import re

from dataclasses import dataclass

@dataclass(frozen = True)
class Tag:
    SUCCESS: str = ''
    INVALID_ARGS: str = 'invalid_args'
    DENIED: str = 'denied'
    EXECUTE_FAILED: str = 'execute_failed'
    DEDUP: str = 'duplicate'
    UNKNOWN_TOOL: str = 'unknown_tool'
    NEED_READ: str = 'need_read'
    STALE: str = 'stale'
    EXISTS: str = 'exists'
    NEED_FULL: str = 'need_full'

@dataclass(frozen = True)
class Level:
    PARTIAL: str = 'partial'
    FULL: str = 'full'

@dataclass(frozen = True)
class Outcome:
    COMPLETED: str = 'completed'
    INTERRUPTED: str = 'interrupted'
    ERROR: str = 'error'
    EXHAUSTED: str = 'exhausted'
    TIMEOUT: str = 'timeout'

TAG = Tag()
LEVEL = Level()
OUTCOME = Outcome()
MARK = '####MINI_HARNESS_RUN####'
MORE = '[Showing lines '
HIT = re.compile(r'^\[(.+?)\]: \d+: ', re.M)