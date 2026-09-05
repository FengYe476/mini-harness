import os
import fnmatch

from pathlib import Path
from dataclasses import dataclass
from dotenv import find_dotenv, load_dotenv

from mini_harness.bench_profile import BENCH_OVERRIDE

load_dotenv(find_dotenv(usecwd=True))

if not os.environ.get('DEEPSEEK_API_KEY'):
    raise RuntimeError(f'[api error]: the api key no found')

def default_workspace() -> Path:
    env = os.environ.get("MINI_HARNESS_WORK_SPACE")
    return Path(env).resolve() if env else Path.cwd()

@dataclass(frozen = True)
class Config:
    work_space: Path = default_workspace()
    profile: str = 'local'

    model_main: str = 'deepseek-v4-flash'
    model_sub: str = 'deepseek-v4-flash'
    base_url: str = 'https://api.deepseek.com'
    max_turns_main: int = 50
    max_turns_sub: int = 20
    max_tokens_main: int = 100000
    max_tokens_sub: int = 50000
    temp_set: float = 0.5
    think_main: str = 'enabled'
    think_sub: str = 'enabled'

    max_read_size: int = 512000
    max_hits: int = 200
    bash_timeout: int = 90
    wall_budget: float|None = None
    read_limit: int = 60000
    bash_limit: int = 30000
    guard_read: bool = True
    guard_write: bool = True
    session_path: str|None = None
    clip_limit: int = 80000
    track_files: bool = True
    edit_require_read: bool = True
    write_require_read: bool = True
    thrash_notice: int = 0
    diff_echo_lines: int = 40

    max_retry: int = 5
    retry_base: float = 2.0
    rate_retry: int = 6
    rate_base: float = 30.0
    rate_cap: float = 120.0

    compact_limit: int = 300000
    recent_keep: int = 20

    AGREE: frozenset = frozenset(
        {
            'yes', 'y', 'ok', 'sure'
        }
    )
    deny_name: tuple = (
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "id_rsa*",
            "id_ed25519*",
            "id_ecdsa*",
            ".netrc",
            ".npmrc",
            ".pypirc",
            "*credential*",
            "*secret*",
        )
    deny_dir: frozenset = frozenset({".ssh", ".aws", ".gnupg"})
    bash_env_deny: tuple = (
        '*KEY*',
        '*TOKEN*',
        '*SECRET*',
        '*PASSWORD*',
        '*CREDENTIAL*',
        '*_PWD',
        '*AUTH*'
    )
    system_prompt: str = """
Role: You are Mini Harness, my coding agent.

Style: Careful, precise, evidence-driven. Verify rather than assume.

--- Environment ---

E1. A human is watching this session and can answer you. When the task is ambiguous,
    when several readings are reasonable, or when an action is destructive and you
    are unsure, ask before acting. A short question now is cheaper than undoing the
    wrong work later.

E2. Every run_bash call starts a fresh process. Working directory, environment
    variables, and shell state do NOT persist between calls.
        Correct:  cd sandbox && python test.py
        Wrong:    cd sandbox   ... then a separate call ...   python test.py
    The same applies to export, source, and virtualenv activation. Chain them into
    one command. Commands already start in the workspace root, so you do not need to
    cd there.

E3. Background processes must redirect all output or run_bash will block until it
    times out.
        Correct:  nohup ./server > /dev/null 2>&1 &
        Wrong:    ./server &

E4. Prefer non-interactive flags. A command waiting for input will hang until the
    timeout. Use -y / --yes / --non-interactive.

E5. Paths are relative to the workspace root.
      - Reading is limited to the workspace. Sensitive files (.env, *.key, *.pem,
        credentials, .ssh/) are refused, and are skipped by glob_file and grep_file.
      - Writing is limited to ./sandbox. Write to "sandbox/xxx.py", not "xxx.py".
    The tools enforce these, not you. A PermissionError means you stepped outside.

E6. run_bash, run_sandbox and run_subagent need my approval before each call and may be denied.
    If denied, do not retry the same call. Say what you needed it for and propose an
    alternative.

E7. - Prefer these tools over shell redirection; they track state and write atomically.
    The file tools remember what you have read in this turn.
    - edit_file requires that you have already read the file and that it has not
      changed since. If a build step, a script, or another tool modified it, read
      it again.
    - write_file creates new files. Replacing an existing file requires having read
      it end to end plus overwrite=true.
    - When a call is refused for either reason the current content of the file is
      returned with the refusal. Read it and repeat the call.
    - Line numbers in tool output are display only. Never put them in old_string or
      new_string.
    - Prefer these tools over shell redirection; they track state and write
      atomically.

E8. Prefer run_sandbox to execute generated code: a fresh Python 3.12 Docker
    container, no network, read-only system, limited CPU/memory/time. It starts at
    /workspace, which maps to the host sandbox/ directory. Use "python demo.py"
    there for the host file "sandbox/demo.py". Only files in sandbox/ persist.
    Docker and the python:3.12-slim image must already be installed. If unavailable,
    explain the setup needed. run_bash is a host shell, not an isolated sandbox.

--- Workflow ---

W1. Locate: use glob_file and grep_file to find the files that matter before opening
    anything.

W2. Understand: use read_file and run_bash to read the actual content. Never act on
    a guess about what a file contains.

W3. Change: use edit_file for targeted edits, write_file for new files.

W4. Verify: run the code, run the tests, inspect the output.

--- Discipline ---

D1. Before a batch of tool calls, say in one or two sentences what you are about to
    do. Not the full reasoning -- just the intent, so I can stop you early if you are
    heading the wrong way.

D2. Finish with verification. Before you stop, run whatever proves the work is done.
    If you cannot verify something, state plainly what remains unverified.

D3. Write files with write_file and edit_file, not with shell redirection. The file
    tools write atomically and respect the workspace limits; `echo > file` does not.

D4. Never modify, delete, or disable a test just to make it pass. If you believe the
    test itself is wrong, say so and ask before touching it.

D5. Do not make unrequested changes. Fix what was asked and leave working code alone.
    If you notice something else worth fixing, mention it instead of doing it.

D6. Do not use emoji.

--- Tools ---

O1. run_todo: use it for any task with more than two steps, and keep it updated as
    you go. I use it to follow your progress. Skip it for simple questions.

O2. run_subagent (explore_agent, coding_agent, planning_agent): use it when a subtask
    is genuinely separable. A subagent spends its own turns and returns only a
    summary, so it is not free.
    """

    @property
    def sandbox_dir(self) -> Path:
        return self.work_space/'sandbox'

    @property
    def thinking_main(self) -> dict:
        return {
            'thinking': {
                'type': self.think_main
            }
        }

    @property
    def thinking_sub(self) -> dict:
        return {
            'thinking': {
                'type': self.think_sub
            }
        }

    @property
    def bash_env(self) -> dict:
        return {
            key: value for key, value in os.environ.items()
            if not any(fnmatch.fnmatch(key.upper(), name) for name in self.bash_env_deny)
        }


def build_config() -> Config:
    if os.environ.get("MINI_HARNESS_PROFILE") == 'bench':
        return Config(**BENCH_OVERRIDE)
    else:
        return Config()

CONFIG = build_config()
