import os
import json
import time

from openai import OpenAI, LengthFinishReasonError
from pathlib import Path
from dataclasses import dataclass, asdict

from mini_harness.config import CONFIG
from mini_harness.retry_request import retry_call
from mini_harness.compact import COMPACT
from mini_harness.tool.box import _ask_human, _always_allow, ToolExecution, _to_api_tool, _atomic_write, log_tool
from mini_harness.tool.tag import OUTCOME, MARK
from mini_harness.tool.block import CLIP

@dataclass(frozen = True)
class Result:
    outcome: str
    calls: int
    turns: int
    ok: int
    failed_by_tag: dict
    calls_by_tool: dict
    last_prompt: int
    prompt_total: int
    completion_total: int
    wall: float
    err: str = ''


class DeepSeekAgent:
    def __init__(self, tools: list, cfg = CONFIG) -> None:
        self.tools = _to_api_tool(tools)
        self.regis = {t.name: t for t in tools}
        self.last_prompt_tokens = 0
        self.printed = ''
        self.system = [
            {
                'role': 'system',
                'content': cfg.system_prompt
            }
        ]
        self.message = list(self.system)
        self.session_memory = cfg.session_path if cfg.session_path else cfg.work_space/'session.json'
        self.last_usage = None
        return

    def _load_memory(self, cfg = CONFIG) -> list:
        if Path(self.session_memory).exists():
            ans = input('\nFound the memory, want to continue the conversation?(yes/no) -> ').strip().lower()
            if ans in cfg.AGREE:
                try:
                    data = json.loads(Path(self.session_memory).read_text(errors = 'replace', encoding = 'utf-8'))
                except (OSError, json.JSONDecodeError) as e:
                    print(f'[load error]: cannot load memory: {e}, start a new conversation')
                    return self.message
                self.message = data
                return self.message
        return self.message

    def _save_memory(self, quiet: bool = False,  cfg = CONFIG) -> None:
        try:
            data = json.dumps(self.message, indent = 2, ensure_ascii=False)
            _atomic_write(Path(self.session_memory), data)
        except (TypeError, ValueError, OSError) as e:
            print(f'[save failed]: cannot save the memory: {e}')
            return

        if not quiet:
            print(f'[save successfully!]')
        return

    def _request_agent(self, client: OpenAI, cfg = CONFIG):
        buffer = ''
        streaming = False
        truncated = None
        with client.chat.completions.stream(
            model = cfg.model_main,
            tools = self.tools,
            messages = self.message,
            temperature = cfg.temp_set,
            extra_body = cfg.thinking_main,
            max_tokens = cfg.max_tokens_main,
            stream_options = {'include_usage': True}
        ) as e:
            for d in e:
                if d.type == 'content.delta':
                    if streaming:
                        print(d.delta, end = '', flush = True)
                        self.printed += d.delta
                        continue

                    buffer += d.delta
                    n = min(len(buffer), len(self.printed))
                    if buffer[:n] != self.printed[:n]:
                        print('\n---- [connection lost, the text above is useless] ----')
                        print(buffer, end = '', flush = True)
                        self.printed = buffer
                        streaming = True
                    elif len(buffer) > len(self.printed):
                        print(buffer[len(self.printed):], end = '', flush = True)
                        self.printed = buffer
                        streaming = True
                if d.type == 'chunk':
                    u = getattr(d.chunk, 'usage', None)
                    if u is not None:
                        self.last_usage = u
                    if d.chunk.choices:
                        res = getattr(d.chunk.choices[0].delta, 'reasoning_content', None)
                        if res:
                            print(res, end = '', flush = True)
                if d.type == 'tool_calls.function.arguments.delta':
                    pass

            try:
                response = e.get_final_completion()
            except LengthFinishReasonError as ex:
                truncated = buffer
                comp = getattr(ex, 'completion', None)
                u = getattr(comp, 'usage', None) if comp is not None else None
                if u is not None:
                    self.last_usage = u
                response = None
        if not streaming and self.printed:
            print(f'\n--- [reconnection lost, the context above is useless] ----')
            if buffer:
                print(buffer, end = '', flush = True)
            self.printed = buffer
        return response, truncated

    def _fill_interrupted(self, tool_calls, cfg = CONFIG) -> None:
        done = [t['tool_call_id'] for t in self.message if t['role'] == 'tool']
        for tool_call in tool_calls:
            if tool_call.id not in done:
                self.message.append(
                    {'role': 'tool', 'tool_call_id': tool_call.id, 'content': f'[intterupted]: the tool {tool_call.function.name} is intterupted'}
                )
        return

    def _run_turn(self, client: OpenAI, executer: ToolExecution, cfg = CONFIG) -> Result:
        start = time.time()
        outcome = OUTCOME.ERROR
        turns = calls = ok = last_prompt = prompt_total = completion_total = 0
        err = ''
        by_tag = {}
        by_tool = {}
        try:
            for turn in range(cfg.max_turns_main):
                if cfg.wall_budget is not None and time.time() -start >= cfg.wall_budget:
                    outcome = OUTCOME.TIMEOUT
                    print(f'[timeout]: agent run out of the time in the task')
                    break
                turns += 1
                self.printed = ''
                if self.last_prompt_tokens >= cfg.compact_limit:
                    self.message = COMPACT.compact_content(client, self.message, self.session_memory, cfg = cfg)
                    self._save_memory(quiet=True)
                    self.last_prompt_tokens = 0
                response, truncated = retry_call(lambda: self._request_agent(client, cfg = cfg), cfg = cfg)
                if truncated is not None:
                    note = ('\n\n[Your previous response was cut off at the output token limit. Be more concise, or take an action instead of continuing to reason.]')
                    self.message.append({
                        'role': 'assistant', 'content': truncated + note
                    })
                    com_degree = self.last_usage
                    if com_degree is not None:
                        self.last_prompt_tokens = getattr(com_degree, 'prompt_tokens', 0) or 0
                        last_prompt = self.last_prompt_tokens
                        prompt_total += self.last_prompt_tokens
                        completion_total += getattr(com_degree, 'completion_tokens', 0) or 0
                    print(f'[ctx]: {last_prompt} / {cfg.compact_limit} tokens, out {getattr(com_degree, 'completion_tokens', 0)} [TRUNCATED]')
                    self._save_memory(quiet=True)
                    continue
                if response.usage:
                    self.last_prompt_tokens = response.usage.prompt_tokens
                    last_prompt = response.usage.prompt_tokens
                    prompt_total += response.usage.prompt_tokens
                    completion_total += response.usage.completion_tokens
                    print(f'[ctx]: {last_prompt} / {cfg.compact_limit} tokens, out {response.usage.completion_tokens}')
                message = response.choices[0].message
                if message.tool_calls:
                    print()
                    d = message.model_dump(exclude_none = True)
                    d.pop('reasoning_content', None)
                    self.message.append(d)
                    try:
                        for tool_call in message.tool_calls:
                            res = executer.execute_tool(tool_call, cfg =cfg)
                            calls += 1
                            by_tool[tool_call.function.name] = by_tool.get(tool_call.function.name, 0) + 1
                            content = CLIP.clip(res.content)
                            if res.ok:
                                ok += 1
                                if tool_call.function.name == 'run_todo':
                                    print(f'\n-=-=-=-=-= Todo List -=-=-=-=-=\n{res.content}')
                            else:
                                by_tag[res.tag] = by_tag.get(res.tag, 0) + 1
                            self.message.append(
                                {'role': 'tool', 'tool_call_id': tool_call.id, 'content': content}
                            ) 
                            log_tool(tool_call, res, cfg = cfg)
                    except KeyboardInterrupt:
                        self._fill_interrupted(message.tool_calls, cfg = cfg)
                        raise
                    self._save_memory(quiet=True)
                else:
                    outcome = OUTCOME.COMPLETED
                    self.message.append(
                        {'role': 'assistant', 'content': message.content}
                    )
                    self._save_memory(quiet=True)
                    break
            else:
                outcome = OUTCOME.EXHAUSTED
                self._save_memory(quiet=True)
                print(f'[agent done]: the agnet run out of the turn for the actions')
        except KeyboardInterrupt:
            outcome = OUTCOME.INTERRUPTED
            print(f'[intterupted]: the action was intterupted')
            self._save_memory(quiet=True)
        except Exception as e:
            outcome = OUTCOME.ERROR
            err = f'{type(e).__name__}:{e}'
            print(f'[run failed]: agent run failed: {err}')
            self._save_memory(quiet=True)
        return Result(
            outcome = outcome,
            calls = calls,
            turns = turns,
            ok = ok,
            last_prompt=last_prompt,
            prompt_total = prompt_total,
            completion_total=completion_total,
            failed_by_tag=by_tag,
            calls_by_tool=by_tool,
            wall = time.time() -start,
            err = err
        )

    def run_task(self, task: str, cfg = CONFIG) -> Result:
        self.message = list(self.system)
        client = OpenAI(
            api_key = os.environ.get("DEEPSEEK_API_KEY"),
            base_url = cfg.base_url,
            max_retries = 0
        ) 
        executer = ToolExecution(self.regis, _always_allow, cfg = cfg)
        self.message.append(
            {'role': 'user', 'content': task}
        )
        result = self._run_turn(client, executer, cfg = cfg)
        self._save_memory(quiet=True)
        return result

    def dump_run(self, result: Result, task: str, path: str|None = None, cfg = CONFIG) -> None:
        payload = {
            'profile': cfg.profile,
            'model': cfg.model_main,
            'thinking': cfg.think_main,
            'task': task,
            'max_turns_main': cfg.max_turns_main,
            **asdict(result)
        }
        print(f'{MARK}{json.dumps(payload, ensure_ascii=False)}')
        if path:
            try:
                p = Path(path)
                _atomic_write(p, json.dumps(payload, indent=2, ensure_ascii=False))
            except OSError as e:
                print(f'[dump failed]: {e}')
        return
                                            
    def run(self, cfg = CONFIG) -> None:
        self.message = self._load_memory(cfg = cfg)
        client = OpenAI(
            api_key = os.environ.get("DEEPSEEK_API_KEY"),
            base_url = cfg.base_url,
            max_retries = 0
        )
        pending_exits = False
        while True:
            try:
                user_input = input("\nPlease enter your command -> ")
            except KeyboardInterrupt:
                if not pending_exits:
                    print(f'Please click ctrl+c again to exit the program ')
                    pending_exits = True
                    continue
                print('\nBye!')
                self._save_memory(quiet = True)
                break
            except EOFError:
                print('\nBye!')
                self._save_memory(quiet = True)
                break

            if user_input.strip().lower() in ['quit', 'exit', 'bye']:
                self._save_memory()
                print('\nBye!')
                break

            self.message.append(
                {'role': 'user', 'content': user_input}
            )
            pending_exits = False
            executer = ToolExecution(self.regis, _ask_human, cfg = cfg)
            result = self._run_turn(client, executer, cfg = cfg)
            self._save_memory(quiet=True)
            print(
                f'[outcome]: {result.outcome}, [calls]: {result.calls}, [turns]: {result.turns}, [ok]: {result.ok}'
                f'[calls_tool]: {result.calls_by_tool}, [failed]: {result.failed_by_tag}'
                f'[last_prompt]: {result.last_prompt}, [last_prompt_tokens]: {result.prompt_total}, [completion_tokens]: {result.completion_total}'
                f'[wall]: {result.wall}'
                f'[err]: {result.err}'
            )
        return
                
