# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "textual==8.2.8", "openai==2.45.0", "httpx==0.28.1",
#   "pydantic==2.13.4", "python-dotenv==1.2.2",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Footer, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

ROOT = Path(__file__).resolve().parent
STORE = ROOT / ".local" / "tui-sessions"
LOGO = "  ╭╴ ╶╮\n  │ ─ │\n  ╰╴ ╶╯\n       "


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def paired_history(messages: list[dict]) -> list[dict]:
    repaired, pending = [], {}
    for message in messages:
        if message.get("role") != "tool" and pending:
            repaired.extend({"role": "tool", "tool_call_id": key, "content":
                             "[interrupted] No result received. Check current files before retrying."}
                            for key in pending)
            pending = {}
        repaired.append(message)
        if message.get("role") == "assistant":
            pending = {call["id"]: True for call in message.get("tool_calls", [])}
        elif message.get("role") == "tool":
            pending.pop(message.get("tool_call_id"), None)
    repaired.extend({"role": "tool", "tool_call_id": key, "content":
                     "[interrupted] No result received. Check current files before retrying."}
                    for key in pending)
    return repaired


def agent_worker() -> None:
    channel = sys.stdout

    def emit(kind: str, **data: object) -> None:
        channel.write(json.dumps({"kind": kind, **data}, ensure_ascii=False) + "\n")
        channel.flush()

    class Output(io.TextIOBase):
        def write(self, text: str) -> int:
            if text and not text.startswith("[ctx]:"):
                emit("text", text=text)
            return len(text)

        def flush(self) -> None:
            channel.flush()

    try:
        request = json.loads(sys.stdin.readline())
        demo = request.get("demo", False)
        os.environ["MINI_HARNESS_PROFILE"] = "local"
        os.environ["MINI_HARNESS_WORK_SPACE"] = str(ROOT)
        if demo:
            os.environ["DEEPSEEK_API_KEY"] = "offline-demo-not-a-real-key"
        sys.path.insert(0, str(ROOT / "src"))
        from openai import OpenAI
        from openai.types.chat import ChatCompletionMessage
        import mini_harness.agent as core
        from mini_harness.config import CONFIG
        from mini_harness.tool import box

        class ConnectedAgent(core.DeepSeekAgent):
            def _save_memory(self, quiet=False, cfg=CONFIG):
                emit("history", messages=self.message, tokens=self.last_prompt_tokens)

            def _request_agent(self, client, cfg=CONFIG):
                self._save_memory()
                if not demo:
                    result = super()._request_agent(client, cfg=cfg)
                else:
                    count = sum(m["role"] == "user" for m in self.message)
                    prompt = next(m["content"] for m in reversed(self.message) if m["role"] == "user")
                    needs_tool = self.message[-1]["role"] == "user"
                    if "what can mini-harness do" in prompt.lower():
                        needs_tool = False
                        text = ("Nine tools, context compaction, and request retries in a small Python "
                                "agent. Give me a task and follow each step as it streams here.")
                    elif "beginner" in prompt.lower() or "learn agents" in prompt.lower():
                        needs_tool = False
                        text = ("Start with agent.py, then explore tool/box.py, compact.py, and "
                                "retry_request.py. Try the offline demo first; add an API key when "
                                "you are ready to run real tasks.")
                    else:
                        text = ("I’ll list the agent files. You can approve or deny this tool call."
                                if needs_tool else f"Demo complete. This conversation contains {count} user "
                                "message(s). Expand the tool card to inspect its input and result.")
                    for word in text.split(" "):
                        print(word + " ", end="", flush=True)
                        time.sleep(0.045)
                    print()
                    calls = [{"id": uuid.uuid4().hex, "type": "function", "function": {
                        "name": "run_bash", "arguments": json.dumps({"command": "ls src/mini_harness"})}}]
                    message = ChatCompletionMessage(role="assistant", content=text,
                                                    tool_calls=calls if needs_tool else None)
                    usage = SimpleNamespace(prompt_tokens=128 * count, completion_tokens=48)
                    result = (SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage), None)
                response, _ = result
                usage = response.usage if response is not None else self.last_usage
                if usage is not None:
                    emit("usage", prompt=usage.prompt_tokens, completion=usage.completion_tokens)
                return result

        tools = list(box.TOOLS)
        if demo:
            tools = [replace(t, function=lambda inp: "agent.py  compact.py  config.py  retry_request.py  tool/\n[Offline simulation]")
                     if t.name == "run_bash" else t for t in tools]
        agent = ConnectedAgent(tools)
        agent.session_memory = Path(request["session_path"])
        history = request.get("messages", [])
        agent.message = paired_history(history) if history else list(agent.system)
        agent.last_prompt_tokens = request.get("tokens", 0)
        agent.message.append({"role": "user", "content": request["prompt"]})

        def confirm(call, cfg=CONFIG):
            emit("confirm", id=call.id, name=call.function.name, arguments=call.function.arguments)
            reply = json.loads(sys.stdin.readline() or "{}")
            return reply.get("id") == call.id and reply.get("allow") is True

        executor = box.ToolExecution(agent.regis, confirm)
        original_execute = box.ToolExecution.execute_tool

        def execute(current, call, cfg=CONFIG):
            if current is executor:
                agent._save_memory()
            emit("tool_start", id=call.id, name=call.function.name,
                 arguments=call.function.arguments, nested=current is not executor)
            result = original_execute(current, call, cfg=cfg)
            emit("tool_end", id=call.id, ok=result.ok, tag=result.tag, content=result.content)
            if current is executor:
                emit("history", messages=[*agent.message, {"role": "tool", "tool_call_id": call.id,
                     "content": core.CLIP.clip(result.content)}], tokens=agent.last_prompt_tokens)
            return result

        box.ToolExecution.execute_tool = execute
        core.log_tool = lambda *args, **kwargs: None
        with contextlib.redirect_stdout(Output()), contextlib.redirect_stderr(Output()):
            with OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=CONFIG.base_url,
                        max_retries=0, timeout=90) as client:
                agent._save_memory()
                result = agent._run_turn(client, executor)
                agent._save_memory()
                emit("done", result=asdict(result))
    except KeyboardInterrupt:
        emit("error", text="Cancelled before the agent finished starting.")
    except Exception as error:
        emit("error", text=f"{type(error).__name__}: {error}")


class ToolConfirmation(ModalScreen[bool]):
    BINDINGS = [("escape", "deny", "Deny")]

    def __init__(self, name: str, arguments: str):
        super().__init__()
        self.tool_name, self.arguments = name, arguments

    def compose(self) -> ComposeResult:
        with Vertical(id="approval"):
            yield Label("PERMISSION REQUIRED", id="approval-heading")
            yield Static(self.tool_name, id="approval-tool", markup=False)
            with VerticalScroll(id="approval-content"):
                yield Static(self.arguments, markup=False)
            yield Static("Allow this tool call once?", id="approval-hint")
            with Horizontal(id="approval-buttons"):
                yield Button("Deny · Esc", id="deny")
                yield Button("Allow once", variant="primary", id="allow")

    def on_mount(self) -> None:
        self.query_one("#deny", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")

    def action_deny(self) -> None:
        self.dismiss(False)


class MiniHarness(App):
    TITLE = "mini-harness"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+n", "new_session", "New"), Binding("f2", "sessions", "Sessions"),
        Binding("ctrl+e", "details", "Tool details", priority=True),
        Binding("ctrl+c", "stop", "Cancel", priority=True),
        Binding("ctrl+q", "quit_cleanly", "Quit", priority=True),
    ]
    CSS = """
    Screen { background: #10161f; color: #d8e2f0; }
    #masthead { height: 6; padding: 0 2; background: #151e2b; border-bottom: solid #29364a; }
    #logo { width: 1fr; height: 4; margin-top: 1; }
    #mode { width: 20; height: 4; content-align: right middle; color: #95a7bd; }
    #body { height: 1fr; }
    #sidebar { width: 26; background: #121b27; border-right: solid #29364a; padding: 1; }
    #sidebar Label { color: #6edbc5; text-style: bold; margin-bottom: 1; }
    #sessions { background: transparent; border: none; height: 1fr; }
    #sessions > .option-list--option-highlighted { background: #263a4c; color: #edfaf7; }
    #main { width: 1fr; }
    #feed { height: 1fr; padding: 1 2; scrollbar-color: #40566d; }
    .message { height: auto; padding: 1 2; margin-bottom: 1; border-left: thick #40566d; }
    .user { background: #1a2938; border-left: thick #6edbc5; }
    .assistant { background: #151e2b; }
    .notice { height: auto; color: #9aacc2; margin: 0 0 1 1; }
    .welcome { height: auto; margin: 2 1; color: #a7b8ce; }
    .tool { height: auto; background: #14232d; margin-bottom: 1; border: round #2c4854; }
    .tool Contents { padding: 0 1 1 1; }
    .tool Static { height: auto; }
    #composer { height: 5; padding: 1 2; background: #151e2b; }
    #prompt { width: 1fr; border: tall #40566d; background: #10161f; }
    #prompt:focus { border: tall #6edbc5; }
    #send, #stop { min-width: 8; width: 9; margin-left: 1; }
    Button { background: #233448; border: none; }
    Button.-primary { background: #285e55; color: #edfaf7; }
    #status { height: 1; padding: 0 2; color: #a7b8ce; background: #1a2635; }
    Footer { background: #121b27; }
    ToolConfirmation { align: center middle; background: #000000 65%; }
    #approval { width: 76; max-width: 95%; height: auto; max-height: 90%;
        padding: 1 2; background: #172433; border: round #6edbc5; }
    #approval-heading { color: #6edbc5; text-style: bold; margin-bottom: 1; }
    #approval-tool { color: #f3d69b; height: auto; margin-bottom: 1; }
    #approval-content { height: auto; max-height: 12; background: #10161f; padding: 1; }
    #approval-hint { height: auto; margin-top: 1; }
    #approval-buttons { height: 3; align-horizontal: right; margin-top: 1; }
    #approval-buttons Button { margin-left: 1; }
    """

    def __init__(self, *, demo: bool = False, sessions_dir: Path = STORE):
        super().__init__()
        self.demo, self.sessions_dir = demo, sessions_dir
        self.session: dict = {}
        self.session_path: Path | None = None
        self.paths: dict[str, Path] = {}
        self.proc: asyncio.subprocess.Process | None = None
        self.runner: asyncio.Task | None = None
        self.busy = self.cancelling = self.expanded = self.dirty = False
        self.live_entry: dict | None = None
        self.live_widget: Static | None = None
        self.cards: dict[str, tuple[Collapsible, Static, dict]] = {}
        self.started = 0.0
        self.phase = "READY"
        self.prompt_tokens = self.output_tokens = self.calls = 0

    def compose(self) -> ComposeResult:
        lines = LOGO.splitlines()
        brand = Text()
        for index, line in enumerate(lines):
            brand.append(line, style="bold #6edbc5")
            brand.append(["   mini-harness", "   SMALL CORE. REAL WORK.",
                          "   A workspace for thinking and building.", ""][index],
                         style="bold #e5edf7" if index == 0 else "#91a6bd")
            if index != 3:
                brand.append("\n")
        with Horizontal(id="masthead"):
            yield Static(brand, id="logo")
            yield Static("OFFLINE DEMO\nno API calls" if self.demo else "LOCAL AGENT\ntool approvals on",
                         id="mode")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("SESSIONS")
                yield OptionList(id="sessions")
            with Vertical(id="main"):
                yield VerticalScroll(id="feed")
                with Horizontal(id="composer"):
                    yield Input(placeholder="What would you like to build?", id="prompt")
                    yield Button("Send", id="send", variant="primary")
                    yield Button("Stop", id="stop", disabled=True)
        yield Static("", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_sessions()
        if self.paths:
            await self.load_session(next(iter(self.paths.values())))
        else:
            await self.action_new_session()
        self.set_interval(0.1, self.refresh_live)
        self.set_interval(2, self.save_session)
        self.fit_layout(self.size.width)
        self.query_one("#prompt", Input).focus()

    def on_resize(self, event) -> None:
        if self.is_mounted:
            self.fit_layout(event.size.width)

    def fit_layout(self, width: int) -> None:
        self.query_one("#sidebar").display = width >= 85
        self.query_one("#mode").display = width >= 75

    def save_session(self) -> None:
        if self.dirty and self.session_path:
            try:
                atomic_json(self.session_path, self.session)
                self.dirty = False
            except OSError as error:
                self.notify(str(error), title="Session could not be saved", severity="error")

    def refresh_sessions(self) -> None:
        self.paths = {}
        options = []
        for path in sorted(self.sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if (data.get("version") != 1 or bool(data.get("demo")) != self.demo
                        or not isinstance(data.get("messages"), list)
                        or not isinstance(data.get("events"), list)):
                    continue
                self.paths[path.stem] = path
                options.append(Option(Text(data.get("title", "Untitled")[:42]), id=path.stem))
            except (OSError, ValueError, AttributeError):
                continue
        widget = self.query_one("#sessions", OptionList)
        widget.clear_options().add_options(options)
        if self.session_path and self.session_path.stem in self.paths:
            widget.highlighted = list(self.paths).index(self.session_path.stem)

    async def action_new_session(self) -> None:
        if self.busy:
            self.notify("Cancel the running task before changing sessions.")
            return
        self.save_session()
        self.session_path = self.sessions_dir / f"{uuid.uuid4().hex[:12]}.json"
        self.session = {"version": 1, "demo": self.demo, "title": "New conversation",
                        "messages": [], "tokens": 0, "events": []}
        self.dirty = True
        self.save_session()
        self.refresh_sessions()
        await self.render_session()

    async def load_session(self, path: Path) -> None:
        if self.busy:
            return
        self.save_session()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data.get("messages"), list) or not isinstance(data.get("events"), list):
                raise ValueError("Invalid session structure")
        except (OSError, ValueError) as error:
            self.notify(str(error), severity="error")
            return
        self.session_path, self.session = path, data
        self.session["messages"] = paired_history(data["messages"])
        await self.render_session()

    async def render_session(self) -> None:
        feed = self.query_one("#feed", VerticalScroll)
        await feed.remove_children()
        self.live_entry = self.live_widget = None
        self.cards.clear()
        self.phase, self.prompt_tokens, self.output_tokens, self.calls = "READY", 0, 0, 0
        if not self.session["events"]:
            await feed.mount(Static("Start with one clear task.\n\n"
                "Read a codebase · investigate a failure · build something small\n\n"
                "Enter to send. Ctrl+N for a new conversation.\n"
                "Tool calls can be expanded; sensitive tools ask before running.", classes="welcome"))
        for entry in self.session["events"]:
            await self.mount_entry(entry)
        self.refresh_live()
        feed.anchor()

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if not self.busy and event.option.id in self.paths:
            await self.load_session(self.paths[event.option.id])

    async def mount_entry(self, entry: dict) -> Static | None:
        feed = self.query_one("#feed", VerticalScroll)
        if entry["kind"] == "tool":
            detail = Static(self.tool_detail(entry), markup=False)
            status = entry.get("status", "INTERRUPTED")
            card = Collapsible(detail, title=f"{status} · {entry['name']}",
                               collapsed=not self.expanded, classes="tool")
            await feed.mount(card)
            self.cards[entry["id"]] = (card, detail, entry)
            return None
        widget = Static(self.message_text(entry), markup=False,
                        classes="notice" if entry["kind"] == "notice" else f"message {entry['kind']}")
        await feed.mount(widget)
        return widget

    @staticmethod
    def message_text(entry: dict) -> Text:
        label = "YOU" if entry["kind"] == "user" else "ASSISTANT"
        text = Text("" if entry["kind"] == "notice" else label + "\n", style="#91a6bd")
        text.append(entry.get("text", "")[-48000:].rstrip(), style="#d8e2f0")
        return text

    @staticmethod
    def tool_detail(entry: dict) -> str:
        return ("ARGUMENTS\n" + entry.get("arguments", "") + "\n\nRESULT\n" +
                entry.get("content", "Waiting for the tool…"))[:24000]

    async def append_entry(self, kind: str, **data) -> Static | None:
        entry = {"kind": kind, **data}
        self.session["events"].append(entry)
        self.dirty = True
        return await self.mount_entry(entry)

    def refresh_live(self) -> None:
        if self.live_widget and self.live_entry:
            self.live_widget.update(self.message_text(self.live_entry))
        elapsed = f" · {time.monotonic() - self.started:.0f}s" if self.busy else ""
        self.query_one("#status", Static).update(
            f"{self.phase}{elapsed}  │  ctx {self.prompt_tokens:,}  ·  out {self.output_tokens:,}"
            f"  ·  tools {self.calls}  │  {self.session.get('title', '')[:36]}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        await self.submit_prompt()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            await self.submit_prompt()
        elif event.button.id == "stop":
            await self.action_stop()

    async def submit_prompt(self) -> None:
        prompt = self.query_one("#prompt", Input)
        text = prompt.value.strip()
        if self.busy or not text:
            return
        if not self.demo and not os.environ.get("DEEPSEEK_API_KEY"):
            self.notify("Export DEEPSEEK_API_KEY in your shell, then restart. Use --demo to try offline.",
                        title="API key needed", severity="warning", timeout=8)
            return
        prompt.value = ""
        self.busy, self.cancelling, self.started = True, False, time.monotonic()
        self.phase, self.output_tokens, self.calls = "STARTING", 0, 0
        self.live_entry = self.live_widget = None
        self.query_one("#send", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        self.query_one("#sessions", OptionList).disabled = True
        if not self.session["messages"]:
            self.session["title"] = text[:64]
        if not self.session["events"]:
            await self.query_one("#feed", VerticalScroll).remove_children()
        await self.append_entry("user", text=text)
        self.save_session()
        self.refresh_sessions()
        self.query_one("#feed", VerticalScroll).anchor()
        self.runner = asyncio.create_task(self.run_agent(text))

    async def handle_event(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "text":
            if not self.live_entry:
                if not event["text"].strip():
                    return
                self.live_widget = await self.append_entry("assistant", text="")
                self.live_entry = self.session["events"][-1]
            self.live_entry["text"] += event["text"]
            self.dirty = True
        elif kind == "history":
            self.session.update(messages=event["messages"], tokens=event.get("tokens", 0))
            self.dirty = True
            self.save_session()
        elif kind == "usage":
            self.prompt_tokens = event["prompt"]
            self.output_tokens += event["completion"]
        elif kind == "tool_start":
            self.refresh_live()
            self.live_entry = self.live_widget = None
            self.calls += 1
            self.phase = "TOOL RUNNING"
            await self.append_entry("tool", id=event["id"], name=event["name"],
                                    arguments=event["arguments"], status="RUNNING")
        elif kind == "tool_end" and event["id"] in self.cards:
            card, detail, entry = self.cards[event["id"]]
            entry.update(content=event["content"], status="OK" if event["ok"] else event["tag"].upper())
            card.title = f"{entry['status']} · {entry['name']}"
            detail.update(self.tool_detail(entry))
            self.dirty, self.phase = True, "THINKING"
        elif kind == "confirm":
            self.phase = "AWAITING APPROVAL"
            approved = False
            if not self.cancelling:
                answer = asyncio.get_running_loop().create_future()
                self.push_screen(ToolConfirmation(event["name"], event["arguments"]),
                                 lambda value: answer.set_result(value) if not answer.done() else None)
                approved = await answer
            if self.proc and self.proc.stdin and not self.proc.stdin.is_closing():
                self.proc.stdin.write((json.dumps({"id": event["id"],
                                      "allow": bool(approved and not self.cancelling)}) + "\n").encode())
                await self.proc.stdin.drain()
            self.phase = "THINKING"
        elif kind == "done":
            result = event["result"]
            self.phase = result["outcome"].upper()
            if result.get("err"):
                await self.append_entry("notice", text=result["err"])
        elif kind == "error":
            self.phase = "ERROR"
            await self.append_entry("notice", text=event["text"])

    async def run_agent(self, prompt: str) -> None:
        try:
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, cwd=ROOT, start_new_session=True,
                limit=64 * 1024 * 1024)
            payload = {"prompt": prompt, "messages": self.session["messages"],
                       "tokens": self.session.get("tokens", 0), "demo": self.demo,
                       "session_path": str(self.session_path)}
            self.proc.stdin.write((json.dumps(payload) + "\n").encode())
            await self.proc.stdin.drain()
            self.phase = "THINKING"
            if self.cancelling:
                os.killpg(self.proc.pid, signal.SIGINT)
            async for line in self.proc.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    await self.append_entry("notice", text=line.decode(errors="replace").strip())
                else:
                    await self.handle_event(event)
            code = await self.proc.wait()
            if code and not self.cancelling:
                self.phase = "ERROR"
                await self.append_entry("notice", text=f"Agent process exited with code {code}.")
        except Exception as error:
            self.phase = "ERROR"
            await self.append_entry("notice", text=f"{type(error).__name__}: {error}")
        finally:
            if self.proc and self.proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                await self.proc.wait()
            self.refresh_live()
            if self.cancelling:
                self.phase = "CANCELLED"
                await self.append_entry("notice", text="Task cancelled. Completed file changes remain.")
            for card, detail, entry in self.cards.values():
                if entry.get("status") == "RUNNING":
                    entry.update(status="INTERRUPTED", content="No final tool result was received.")
                    card.title = f"INTERRUPTED · {entry['name']}"
                    detail.update(self.tool_detail(entry))
            self.session["messages"] = paired_history(self.session["messages"])
            self.busy, self.proc, self.dirty = False, None, True
            self.query_one("#send", Button).disabled = False
            self.query_one("#stop", Button).disabled = True
            self.query_one("#sessions", OptionList).disabled = False
            self.save_session()
            self.refresh_sessions()
            self.refresh_live()
            self.query_one("#prompt", Input).focus()

    async def action_stop(self) -> None:
        if not self.busy or self.cancelling:
            return
        self.cancelling, self.phase = True, "CANCELLING"
        if isinstance(self.screen, ToolConfirmation):
            self.screen.dismiss(False)
        proc = self.proc
        if not proc:
            return
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, sig)
            try:
                await asyncio.wait_for(proc.wait(), timeout=6 if sig == signal.SIGINT else 0.7)
                break
            except asyncio.TimeoutError:
                continue
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)

    def action_sessions(self) -> None:
        panel = self.query_one("#sidebar")
        panel.display = not panel.display
        if panel.display:
            self.query_one("#sessions", OptionList).focus()

    def action_details(self) -> None:
        self.expanded = not self.expanded
        for card, _, _ in self.cards.values():
            card.collapsed = not self.expanded

    async def action_quit_cleanly(self) -> None:
        await self.action_stop()
        if self.runner and not self.runner.done():
            await self.runner
        self.save_session()
        self.exit()

    async def on_unmount(self) -> None:
        await self.action_stop()
        if self.runner and not self.runner.done():
            with contextlib.suppress(Exception):
                await self.runner
        self.save_session()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Try the interface without an API key or model calls")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        agent_worker()
    else:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        MiniHarness(demo=args.demo).run()


if __name__ == "__main__":
    main()
