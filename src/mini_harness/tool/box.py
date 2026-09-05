import os
import fnmatch
import re
import glob
import subprocess
import json
import time
import hashlib
import shutil
import threading
import uuid

from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Annotated
from pydantic import BaseModel, Field, ConfigDict, ValidationError, StringConstraints, field_validator, model_validator
from enum import Enum
from openai import OpenAI

from mini_harness.config import CONFIG
from mini_harness.tool.path import validate_read, validate_write, is_denied, _resolve_file
from mini_harness.tool.block import TODO, CLIP, SUBAGENT
from mini_harness.retry_request import retry_call
from mini_harness.tool.tag import TAG, MORE, LEVEL, HIT


Nonblank = Annotated[str, StringConstraints(strip_whitespace = True, min_length = 1)]
LINE_NO = re.compile(r'^\s*\d+\t')
READ_TOOLS = {'read_file', 'grep_file'}
WRITE_TOOLS = {'write_file', 'edit_file'}

def _safe_time(p):
    try:
        return p.stat().st_mtime
    except (OSError):
        return 0.0

def _split_lines(content: str) -> list[str]:
    if not content:
        return []

    content = content.replace('\r\n', '\n')
    content_split = content.split('\n')
    if content_split and not content_split[-1]:
        content_split.pop()
    return content_split

def _render_lines(val_path: Path, offset: int|None, limit: int|None, cfg = CONFIG) -> tuple[str, bool]:
    try:
        content = val_path.read_text(encoding = 'utf-8')
    except UnicodeDecodeError:
        size = val_path.stat().st_size
        return f'[Binary or non-UTF-8 file, {size} bytes. Content not displayed]', True
    lines = _split_lines(content)
    total = len(lines)
    if total == 0:
        return f'[The file is empty (0 lines)]', True

    start = (offset if offset else 1) - 1
    end = start + limit if limit else total
    window = lines[start: end]
    if not window:
        raise ValueError(f'[offset error]: offset {offset} exceeds the file length ({total} lines), use an offset between 1 and {total}')

    out, used = [], 0
    for lineno, line in enumerate(window, start + 1): 
        render = f'{lineno:>6}\t{line}'
        out.append(render)
        used += len(render)
        if used >= cfg.read_limit:
            break

    last = start + len(out)
    body = '\n'.join(out)
    reach_end = (last >= total)
    if not reach_end:
        body += f'\n\n{MORE}{start+1}-{last} of {total}, use offset {last + 1} to continue]'

    return body, reach_end
        
def _atomic_write(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents = True, exist_ok=True)
    temp = file_path.with_name(file_path.name + '.tmp')
    try:
        temp.write_text(content, errors = 'replace', encoding = 'utf-8')
        os.replace(temp, file_path)
    finally:
        temp.unlink(missing_ok=True)

def _strip_line_no(content: str) -> str:
    lines = _split_lines(content)
    if not lines:
        return content
    if all(LINE_NO.match(line) for line in lines if line.strip()):
        return '\n'.join(LINE_NO.sub('', line) for line in lines)
    return content

def _key(file_path, cfg = CONFIG) -> str:
    return str(_resolve_file(file_path, cfg = cfg))

def _digest(p: Path) -> str:
    h = hashlib.md5()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def _to_api_tool(tools: list) -> list[dict]:
    return [
        {
            'type': 'function',
            'function': {
                'name': t.name,
                'description': t.description,
                'parameters': t.parameters.model_json_schema()
            }
        } for t in tools
    ]

def _ask_human(tool_call, cfg = CONFIG) -> bool:
    ans = input(f'\nthe agent will use the tool: {tool_call.function.name}: {tool_call.function.arguments}\nALLOW or Not(yes/no) -> ').strip().lower()
    if ans not in cfg.AGREE:
        return False
    return True

def _for_sub(tool_call, cfg = CONFIG) -> bool:
    return False

def _always_allow(tool_call, cfg = CONFIG) -> bool:
    return True

@dataclass
class FileRecord:
    mtime: float
    digest: str
    level: str
    edits: int = 0

@dataclass(frozen = True)
class ToolItem:
    content: str
    ok: bool
    tag: str = ''

class ToolExecution:
    def __init__(self, regis: dict, confirm: Callable, cfg = CONFIG) -> None:
        self.regis = regis
        self.confirm = confirm
        self.last_tool = None
        self.files = {}
        return

    def _mark(self, key: str, level: str) -> None:
        p = Path(key)
        try:
            mtime = p.stat().st_mtime
            digest = _digest(p)
        except OSError:
            return

        old = self.files.get(key)
        if level == LEVEL.FULL or (old and old.level == LEVEL.FULL):
            lvl = LEVEL.FULL
        else:
            lvl = LEVEL.PARTIAL
        edits = old.edits if old else 0
        self.files[key] = FileRecord(mtime, digest, lvl, edits)
        return

    def _fresh(self, key: str) -> bool:
        rec = self.files.get(key)
        if rec is None:
            return False

        p = Path(key)
        try:
            if p.stat().st_mtime == rec.mtime:
                return True
            return _digest(p) == rec.digest
        except OSError:
            return False

    def _deny(self, key: str, tag: str, msg: str, cfg = CONFIG) -> ToolItem:
        try:
            body, reach_end = _render_lines(Path(key), None, None, cfg = cfg)
            self._mark(key, LEVEL.FULL if reach_end else LEVEL.PARTIAL)
            return ToolItem(f'[{tag}]: {msg}\n\nCurrent content of {key}:\n{body}', False, tag)
        except Exception as e:
            return ToolItem(f'[{tag}]: {msg}\n\n[Counld not display the file: {type(e).__name__}. '
                            f'Read it with read_file before before retrying]', False, tag)
    
    def _gate(self, name: str, args, cfg = CONFIG) -> ToolItem|None:
        if not cfg.track_files:
            return None
        if name not in WRITE_TOOLS:
            return None

        file_path = getattr(args, 'file_path', None)
        if file_path is None:
            return None

        key = _key(file_path, cfg = cfg)
        exists = Path(key).is_file()

        if name == 'edit_file' and cfg.edit_require_read:
            if not exists:
                return None
            if key not in self.files:
                return self._deny(key, TAG.NEED_READ, 'You have not read this file in this session. Read it first, then retry')
            if not self._fresh(key):
                return self._deny(key, TAG.STALE, 'This file changed after you last read it. Review the current content below, then retry.')

        if name == 'write_file' and cfg.write_require_read:
            if not exists:
                return None
            if not args.overwrite:
                return self._deny(key, TAG.EXISTS, 'This file already exists. write_file creates new file. To replace it entirely, read it in full first and pass overwrite = true; to change part of it, use edit_file')

            rec = self.files.get(key)
            if rec is None or rec.level != LEVEL.FULL:
                return self._deny(key, TAG.NEED_FULL, 'Overwriting destroys the whole file, so you must have read it end to end first. The current content is below; retry after reviewing it')
            if not self._fresh(key):
                return self._deny(key, TAG.STALE, 'This file changed after you last read it. Review the current content below, then retry')

        return None

    def _record(self, name: str, args, content: str, cfg = CONFIG) -> str:
        if not cfg.track_files:
            return content

        if name == 'read_file':
            key = _key(args.file_path, cfg = cfg)
            self._mark(key, LEVEL.PARTIAL if MORE in content else LEVEL.FULL)

        elif name == 'grep_file':
            for hit in set(HIT.findall(content)):
                self._mark(hit, LEVEL.PARTIAL)

        elif name in WRITE_TOOLS:
            key = _key(args.file_path, cfg = cfg)
            old = self.files.get(key)
            edits = old.edits if old else 0
            level = LEVEL.FULL if name == 'write_file' else (old.level if old else LEVEL.FULL)

            self._mark(key, level)
            self.files[key].edits = edits + 1

            n = self.files[key].edits
            if cfg.thrash_notice and n >= cfg.thrash_notice:
                content += (f'\n\n[You have modified this file {n} times without the task passing].\nConsider re-reading it in full, or reconsidering the approach')
        return content
                
    def execute_tool(self, tool_call, cfg = CONFIG) -> ToolItem:
        tool = self.regis.get(tool_call.function.name)
        if tool is None:
            return ToolItem(f'[{TAG.UNKNOWN_TOOL}]: the tool is unknown, please check the tool map {'\n'.join(self.regis)}', False, TAG.UNKNOWN_TOOL)

        try:
            args = tool.parameters.model_validate_json(tool_call.function.arguments)
        except ValidationError as e:
            return ToolItem(f'[{TAG.INVALID_ARGS}]: cannot unpack the json format: {e}', False, TAG.INVALID_ARGS)

        funs = tool.function
        current = (tool_call.function.name, args.model_dump_json())
        if self.last_tool == current:
            return ToolItem(f'[{TAG.DEDUP}]: same tool {tool_call.function.name} and {tool_call.function.arguments} arguments used, please use the other tool or arguments', False, TAG.DEDUP)
        
        gate = self._gate(tool_call.function.name, args, cfg = cfg)
        if gate is not None:
            return gate

        if tool.risky:
            if not self.confirm(tool_call, cfg = cfg):
                return ToolItem(f'[{TAG.DENIED}]: the use of tool was denied by the user, please tell the user about this situation', False, TAG.DENIED)
        self.last_tool = current
        
        try:
            result = funs(args)
        except Exception as e:
            return ToolItem(f'[{TAG.EXECUTE_FAILED}]: the tool execute failed: {type(e).__name__}: {e}', False, f'{TAG.EXECUTE_FAILED}:{type(e).__name__}')

        raw = result if isinstance(result, str) else json.dumps(result)
        raw = self._record(tool_call.function.name, args, raw, cfg = cfg)
        return ToolItem(raw, True, TAG.SUCCESS)

def log_tool(tool_call, res: ToolItem, prefix: str = '', cfg = CONFIG) -> None:
    arguments = tool_call.function.arguments
    if len(arguments) > 100:
        arguments = arguments[:100] + '\n....clipped at 100 chars'
    if res.ok:
        print(f'{prefix}{tool_call.function.name}: {arguments}')
    else:
        print(f'{prefix}{tool_call.function.name}: {arguments} -> Failed {res.tag}')
    return 

class GlobFileInput(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    pattern: Nonblank = Field(description = 'the pattern of file, the tool will glob the related file in all subdirectory if you input the pattern without "/" (e.g. "*.py" will search all the python file in the subdirectory). you can use "/" to search in the targeted directory (e.g. "src/*.py")')
    path: str = Field('.', description = 'the path which you want to glob')

def glob_file(inp: GlobFileInput, cfg = CONFIG) -> str:
    val_path = validate_read(Path(inp.path), cfg = cfg)
    pattern = inp.pattern
    if '/' not in pattern:
        pattern = '**/' + pattern

    matches = []
    for rel in glob.glob(pattern, root_dir = val_path, recursive = True):
        full = val_path/rel
        real = full.resolve()

        if not real.is_file():
            continue
        if cfg.guard_read and not real.is_relative_to(cfg.work_space):
            continue
        if is_denied(real, cfg = cfg):
            continue

        matches.append(full)
    matches = sorted(set(matches), key = _safe_time, reverse = True)
    return '\n'.join(str(m) for m in matches) if matches else 'no matches'


class GrepFileInput(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    pattern: Nonblank = Field(description = 'the pattern which you want to grep(targeted information)')
    path: str = Field('.', description = 'the path which you want to grep')
    glob: str|None = Field(None, description = 'the targeted type of file which you want to grep')

    @field_validator('pattern')
    @classmethod
    def test_regex(cls, valid: Nonblank) -> Nonblank:
        try:
            re.compile(valid)
        except re.error as e:
            raise ValueError(f'[re error]: invalid regex: {e}')
        return valid

def grep_file(inp: GrepFileInput, cfg = CONFIG) -> str:
    val_path = validate_read(Path(inp.path), cfg = cfg)
    hits = []
    truncated = False
    regex = re.compile(inp.pattern)
    if val_path.is_file():
        target = [val_path]
    else:
        target = []
        for root, dirs, files in os.walk(val_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in cfg.deny_dir]

            for name in files:
                full = Path(root)/name
                real = full.resolve()

                if cfg.guard_read and not real.is_relative_to(cfg.work_space):
                    continue
                if is_denied(real, cfg = cfg):
                    continue
                if inp.glob and not fnmatch.fnmatch(name, inp.glob):
                    continue
                target.append(real)
    for file_path in target:
        try:
            if truncated:
                break
            with open(file_path, 'r', errors = 'replace', encoding = 'utf-8') as f:
                for lineno, lines in enumerate(f, 1):
                    if regex.search(lines):
                        hits.append(f'[{file_path}]: {lineno}: {lines.rstrip()}')
                        if len(hits) >= cfg.max_hits:
                            truncated = True
                            break
        except (OSError, UnicodeDecodeError):
            continue
        if truncated:
            hits.append(f'\n..... truncated at {cfg.max_hits}')
    return '\n'.join(hits) if hits else 'no matches'


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    file_path: Nonblank = Field(description = 'the file path which you want to read')
    offset: int|None = Field(None, ge = 1, description = '1-based line number to start reading from. Same coordinate system as the line numbers in grep_file output and in read_file output.')
    limit: int|None = Field(None, ge = 1, description = 'The number of line which you want to read')

def read_file(inp: ReadFileInput, cfg = CONFIG) -> str:
    val_path = validate_read(Path(inp.file_path), cfg = cfg)
    if not val_path.is_file():
        raise FileNotFoundError(f'[file no found]: This is no file')
    blind = (inp.offset is None and inp.limit is None)
    if blind:
        size = val_path.stat().st_size
        if size >= cfg.max_read_size:
            raise ValueError(f'[oversize]: the size({size} bytes) of {val_path} is too large to read, the max limit is {cfg.max_read_size} bytes')

    body, _ = _render_lines(val_path, inp.offset, inp.limit, cfg = cfg)
    return body

class WriteFileInput(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    file_path: Nonblank = Field(description = 'the file path which you want to create and write')
    content: str = Field(description = 'the content which you want to write')
    overwrite: bool = Field(False, description = 'Set true only when you intend to replace the entire existing file. Requires having read the file in full first. Leave false to create a new file')

def write_file(inp: WriteFileInput, cfg = CONFIG) -> str:
    val_path = validate_write(Path(inp.file_path), cfg = cfg)
    existed = val_path.is_file()
    _atomic_write(val_path, inp.content)
    n = len(_split_lines(inp.content))
    verb = 'Overwrote' if existed else 'Created'
    return f'{verb} {val_path} ({n} lines, {len(inp.content)} chars)'

class EditFileInput(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    file_path: Nonblank = Field(description = 'the file path which you want to edit')
    old_string: str = Field(description = 'The exact text to replace. Must appear in the file verbatim, including indentation. Do not include the line-number prefixes shown by read_file')
    new_string: str = Field(description = 'The replacement text. Do not include line-number prefixes.')
    replace_all: bool = Field(False, description = 'depend on whether you want to replace all the old content into new content (in case the old content appears couple times)')

    @model_validator(mode = 'after')
    def test_content(self):
        if self.old_string == self.new_string:
            raise ValueError(f'[content error]: the old content is equal to the new content')
        return self

def edit_file(inp: EditFileInput, cfg = CONFIG) -> str:
    val_path = validate_write(Path(inp.file_path), cfg = cfg)
    if not val_path.is_file():
        raise FileNotFoundError(f'[file no found]: this is no file')

    try:
        content = val_path.read_text(encoding = 'utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f'[encoding]: this file is not valid UTF-8 (byte {e.start}: {e.reason}.)\nedit_file would corrupt it, because it rewrites the file as UTF-8 text.')
    old = inp.old_string
    if old not in content:
        stripped = _strip_line_no(old)
        if stripped != old and stripped in content:
            if any(LINE_NO.match(line) for line in _split_lines(inp.new_string)):
                raise ValueError(f'[line number]: the old_string contained cat -n line numbers and was stripped, '
                                 'but the new_string contains them too. Line numbers are display only, '
                                 'resend both without the "<number>\\t" prefix')
            if stripped == inp.new_string:
                raise ValueError(f'[content error]: the old content is equal to new content')
            old = stripped
        else:
            raise ValueError(f'[content error]: the old content is not in {val_path}')

    if not inp.replace_all and content.count(old) > 1:
        raise ValueError(f'[content error]: the old content appears in the {val_path} {content.count(old)} times')

    idx = content.index(old)
    start_line = content[:idx].count('\n') + 1

    old_n = len(_split_lines(old))
    new_n = len(_split_lines(inp.new_string))
    hits = content.count(old) if inp.replace_all else 1

    content_new = content.replace(old, inp.new_string) if inp.replace_all else content.replace(old, inp.new_string, 1)
    _atomic_write(val_path, content_new)

    total_new = len(_split_lines(content_new))
    head = (f'Replaced {hits} occurrence(s) in {val_path}\n({old_n} -> {new_n} lines, file now {total_new} lines).')

    if cfg.diff_echo_lines == 0:
        return head

    shown = _split_lines(inp.new_string)[:cfg.diff_echo_lines]
    body = '\n'.join(f'{start_line + lineno:>6}\t{line}' for lineno, line in enumerate(shown))
    if new_n > cfg.diff_echo_lines:
        body += f'\n[showing first {cfg.diff_echo_lines} of {new_n} changed lines]'

    return head + '\n' + body


class RunBashInput(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    command: Nonblank = Field(description = 'the command used to operate the terminal')

def run_bash(inp: RunBashInput, cfg = CONFIG) -> str:
    try:
        result = subprocess.run(
            inp.command,
            shell = True,
            capture_output=True,
            text=True,
            timeout = cfg.bash_timeout,
            env = cfg.bash_env,
            cwd = cfg.work_space
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f'[timeout]: the terminal time out of {cfg.bash_timeout}s: {e}')

    part = []
    if result.stdout:
        part.append(result.stdout)
    if result.stderr:
        part.append(f'[stderr]: {result.stderr}')
    if result.returncode != 0:
        part.append(f'[exit code: {result.returncode}]')
    content = '\n'.join(part) if part else 'no result'
    if len(content) >= cfg.bash_limit:
        origin = len(content)
        content = content[:cfg.bash_limit] + f'\n[.....truncated at {cfg.bash_limit} of {origin}, , Narrow the command (grep/head/tail) or redirect to a file and read it with offset/limit]'  
    return content  

class RunSandboxInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    command: Nonblank = Field(description='Shell command inside an isolated Python container. sandbox/ is mounted at /workspace; use paths relative to it.')
    timeout_seconds: int = Field(default=30, ge=1, le=300, strict=True, description='Maximum command runtime in seconds.')


def run_sandbox(inp: RunSandboxInput, cfg = CONFIG) -> str:
    docker = shutil.which('docker')
    if not docker:
        raise RuntimeError('run_sandbox requires Docker. Install/start Docker, then run: docker pull python:3.12-slim')
    root = cfg.work_space.resolve()
    workspace = cfg.sandbox_dir
    if workspace.is_symlink() or not workspace.resolve().is_relative_to(root):
        raise PermissionError('sandbox/ must be a real directory inside the workspace')
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve()
    if ',' in str(workspace):
        raise ValueError('Docker sandbox paths cannot contain commas')
    name = 'mini-harness-' + uuid.uuid4().hex
    runner = (
        'import subprocess,sys\n'
        'try:\n'
        ' p=subprocess.run(sys.argv[1],shell=True,timeout=int(sys.argv[2]))\n'
        ' sys.exit(p.returncode if p.returncode >= 0 else 128-p.returncode)\n'
        'except subprocess.TimeoutExpired:\n'
        ' print("[sandbox command timed out]",flush=True)\n'
        ' sys.exit(124)\n'
    )
    command = [
        docker, 'run', '--rm', '--pull=never', '--name', name,
        '--label', 'mini-harness.sandbox=true', '--network=none', '--read-only',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=64',
        '--memory=256m', '--memory-swap=256m', '--cpus=1', '--log-driver=none',
        '--ulimit', 'fsize=16777216:16777216', '--user', f'{os.getuid()}:{os.getgid()}',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=64m,mode=1777',
        '--mount', f'type=bind,src={workspace},dst=/workspace', '--workdir', '/workspace',
        '--env', 'HOME=/tmp', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--entrypoint', 'python', 'python:3.12-slim', '-u', '-c', runner,
        inp.command, str(inp.timeout_seconds),
    ]
    output = bytearray()
    clipped = False
    def drain(pipe):
        nonlocal clipped
        with pipe:
            while chunk := pipe.read(8192):
                remaining = max(0, cfg.bash_limit - len(output))
                output.extend(chunk[:remaining])
                clipped |= len(chunk) > remaining

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, env=cfg.bash_env, start_new_session=True)
    reader = threading.Thread(target=drain, args=(process.stdout,), daemon=True)
    reader.start()
    try:
        process.wait(timeout=inp.timeout_seconds + 10)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError('Docker sandbox startup or execution timed out') from error
    finally:
        try:
            cleanup = subprocess.run([docker, 'rm', '-f', name], capture_output=True,
                                     timeout=5, env=cfg.bash_env)
            if cleanup.returncode and b'No such container' not in cleanup.stderr:
                raise RuntimeError(f'Could not remove sandbox container {name}: ' + cleanup.stderr.decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            reader.join(timeout=1)
    text = output.decode('utf-8', errors='replace') or 'no result'
    if clipped:
        text += f'\n[output truncated at {cfg.bash_limit} bytes]'
    if process.returncode:
        if process.returncode == 124:
            raise TimeoutError(f'Sandbox exceeded {inp.timeout_seconds}s.\n{text}')
        raise RuntimeError(f'Sandbox exited with code {process.returncode}.\n{text}')
    return text


class StatusItem(str, Enum):
    completed = 'completed'
    pending = 'pending'
    in_progress = 'in_progress'

class TodoItem(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    content: Nonblank = Field(description = 'the content of the task in todo list')
    activeForm: Nonblank = Field(description = 'the active form of the content in the task, (e.g. fixing the bug)')
    status: StatusItem = Field(description = 'the status of the task, including pending, in_progress and completed')

class RunTodoInput(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    items: list[TodoItem] = Field(description = 'the todo list', min_length = 1, max_length = 20)

    @model_validator(mode = 'after')
    def test_status(self):
        in_progress_count = sum(1 for t in self.items if t.status == 'in_progress')
        if in_progress_count > 1:
            raise ValueError(f'[invalid status]: there is only one thing can be in progress')
        return self

def run_todo(inp: RunTodoInput, cfg = CONFIG) -> str:
    return TODO.update(inp.items)

class AgentType(str, Enum):
    explore_agent = 'explore_agent'
    coding_agent = 'coding_agent'
    planning_agent = 'planning_agent'

class RunSubAgentInput(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    task_description: Nonblank = Field(description = 'the description of task')
    prompt: Nonblank = Field(description = 'the user prompt which you want to give subagent')
    agent_type: AgentType = Field(description = 'the agent type which you want use, including explore_agent(explore, read and find the file and file content based on the task), coding_agent (create, code and program the file based on the task) and planning_agent (read, analyze and make a plan to better finish the task)')

def run_subagent(inp: RunSubAgentInput, cfg = CONFIG) -> str:
    full_tool_list = _to_api_tool(TOOLS)
    config = SUBAGENT.agent_table.get(inp.agent_type.value)
    if config is None:
        raise ValueError(f'[invalid subagent]: invalid subagent {inp.agent_type.value}')
    tool_list = [t for t in full_tool_list if t['function']['name'] in config['tools']]
    regis = {t.name: t for t in TOOLS if t.name in config['tools']}
    tool_count = 0
    sub_message = [
        {
            'role': 'system',
            'content': f'''
You are my subagent-{inp.agent_type.value}, your responsibility is to {config['description']}
{config['prompt']}
completed the task by giving the summary and analyzing report
'''
        },
        {
            'role': 'user',
            'content': inp.prompt
        }
    ]
    start = time.time()
    print(f'[{inp.agent_type.value}]: {inp.task_description}')
    client = OpenAI(
        api_key = os.environ.get("DEEPSEEK_API_KEY"),
        base_url = cfg.base_url,
        max_retries = 0
    )
    executer = ToolExecution(regis, _for_sub, cfg = cfg)
    for turn in range(cfg.max_turns_sub):
        response = retry_call(lambda: client.chat.completions.create(
            model = cfg.model_sub,
            messages = sub_message,
            tools = tool_list,
            max_tokens = cfg.max_tokens_sub,
            extra_body = cfg.thinking_sub,
            temperature = cfg.temp_set,
            stream = False
        ), cfg = cfg)
        message = response.choices[0].message
        if message.tool_calls:
            d = message.model_dump(exclude_none = True)
            sub_message.append(d)
            for tool_call in message.tool_calls:
                res = executer.execute_tool(tool_call, cfg = cfg)
                content = CLIP.clip(res.content, cfg = cfg)
                tool_count += 1
                sub_message.append(
                    {'role': 'tool', 'tool_call_id': tool_call.id, 'content': content}
                )
        else:
            sub_message.append(
                message.model_dump(exclude_none=True)
            )
            end = time.time() -start
            print(f'[{inp.agent_type.value}]: {inp.task_description} -- {tool_count} tools -- {end:.1f}s')
            return message.content
    else:
        return f'[agent done]: the agent run out of the turns for the actions, the task is incompleted'
    

@dataclass(frozen = True)
class ToolDefinition:
    name: str
    description: str
    parameters: type[BaseModel]
    function: Callable
    risky: bool

TOOLS = [
    ToolDefinition('glob_file', 'glob and check the file', GlobFileInput, glob_file, False),
    ToolDefinition('grep_file', 'grep and search for the targeted information', GrepFileInput, grep_file, False),
    ToolDefinition('read_file', 'Read a file. Output uses cat -n style line numbers, which are for reference only and must never be copied into edit_file. Without offset/limit the whole file is returned, truncated with a continuation hint if it is very large.', ReadFileInput, read_file, False),
    ToolDefinition('write_file', 'Create a new file. To replace an existing file entirely you must have read it in full first and pass overwrite=true; for partial changes use edit_file instead.', WriteFileInput, write_file, False),
    ToolDefinition('edit_file', 'Replace a specific string in an existing file. You must have read the file first, and it must not have changed since. old_string must match the file exactly and must not contain line-number prefixes. On success the tool echoes the resulting lines with their real line numbers.', EditFileInput, edit_file, False),
    ToolDefinition('run_bash', 'operate the terminal by using the command', RunBashInput, run_bash, True),
    ToolDefinition('run_sandbox', 'Run code in a disposable Docker Python 3.12 container with no network and resource limits. Only sandbox/ is shared, as /workspace; writes there persist. Requires Docker and a locally pulled python:3.12-slim image. Prefer this for running generated code; run_bash executes on the host.', RunSandboxInput, run_sandbox, True),
    ToolDefinition('run_todo', 'build and update the todo list', RunTodoInput, run_todo, False),
    ToolDefinition('run_subagent', 'build and run the subagent to finish the task', RunSubAgentInput, run_subagent, True)
]

_names = {t.name for t in TOOLS}
if not (READ_TOOLS | WRITE_TOOLS) <= _names:
    raise RuntimeError(f'[tool name drift]: {(READ_TOOLS | WRITE_TOOLS) - _names} not in TOOLS')








    
    
        
    

    

