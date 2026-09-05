<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img src="assets/logo-light.svg" alt="mini-harness" width="420">
  </picture>
</div>

<p align="center">
  <img src="assets/demo.gif" alt="Real DeepSeek Agent writes a Python file, runs it in a Docker sandbox, and answers a follow-up question" width="900">
  <br>
  <sub>Real DeepSeek V4 Flash · 12.8 s replay · <a href="assets/demo.mp4">120 fps video</a></sub>
</p>

## Why mini-harness?

- **Small, but complete.** About 1,700 lines of Python: nine tools, context
  compaction, request retries, streaming responses, and session memory.
- **Tools defined with Pydantic.** Typed inputs, generated JSON Schema, and
  validation before execution make tools easier to compose and orchestrate.
- **A practical baseline.** Evaluated on SWE-bench Verified and Terminal-Bench
  2.1 with DeepSeek V4 Flash. See the results below.
- **Built for learning.** Follow the [agent loop](src/mini_harness/agent.py),
  [tools](src/mini_harness/tool/box.py), [compaction](src/mini_harness/compact.py),
  and [retry policy](src/mini_harness/retry_request.py) in ordinary Python.
  The optional [TUI](tui.py) is a single file you can read and modify.

## Benchmarks

Historical, self-reported results with **DeepSeek V4 Flash**. The latest source
changes have not been re-evaluated on these benchmarks.

| Benchmark | Solved / attempts | Score |
| --- | ---: | ---: |
| SWE-bench Verified | 401 / 500 | **80.2%** |
| Terminal-Bench 2.1 | 309 / 445 | **69.44%** |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/terminal-bench-ranking-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/terminal-bench-ranking-light.svg">
    <img src="assets/terminal-bench-ranking-light.svg" alt="Terminal-Bench 2.1 score comparison: mini-harness at 69.44% would place 13th among 18 public entries plus this self-reported run, as of September 4, 2026. This is not an official rank." width="900">
  </picture>
</p>

The figure inserts our score into the [public Terminal-Bench 2.1 leaderboard](https://www.tbench.ai/?version=2.1)
as of **September 4, 2026**; **13th of 19 is an illustrative position, not an official
rank**. The Terminal-Bench result averages five attempts on each of 89 tasks,
not pass@5. See [evaluation details, recorded failures, and reproduction limits](BENCHMARKS.md).

## Define a tool with Pydantic

A tool combines a Pydantic input model, a Python function, and a `ToolDefinition`.
After installing and exporting `DEEPSEEK_API_KEY` (see [Get started](#get-started)),
pass your definitions to the agent to choose which tools it can use:

```python
from pydantic import BaseModel, ConfigDict, Field

from mini_harness.agent import DeepSeekAgent
from mini_harness.tool.box import TOOLS, ToolDefinition


class CountWordsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, description="Text to count words in.")


def count_words(args: CountWordsInput) -> str:
    return str(len(args.text.split()))


count_words_tool = ToolDefinition(
    name="count_words",
    description="Count whitespace-separated words in text.",
    parameters=CountWordsInput,
    function=count_words,
    risky=False,
)

agent = DeepSeekAgent([*TOOLS, count_words_tool])
```

The agent uses `model_json_schema()` to describe each tool to the model.
Before dispatch, the [tool executor](src/mini_harness/tool/box.py) calls
`model_validate_json()` to validate its arguments; invalid calls return an error
for the agent to correct. One definition keeps the schema, validation, and
execution connected. See the [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/).

## Get started

### 1. Download and install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not
have it, then clone the repository. Python 3.12+ is required; uv can install it.

```sh
# macOS / Linux — install uv once, then restart your terminal
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```sh
git clone https://github.com/mini-harness/mini-harness.git
cd mini-harness
uv python install 3.12
uv sync --locked
```

### 2. Connect the agent

Set your [DeepSeek API key](https://platform.deepseek.com/api_keys) in the shell,
then launch the TUI. No `.env` file is needed.

```sh
export DEEPSEEK_API_KEY="your-api-key"
uv run tui.py
```

`Enter` sends a message · `Ctrl+N` starts a session · `F2` opens sessions ·
`Ctrl+E` expands tools · `Ctrl+C` cancels · `Ctrl+Q` quits.

The agent reads the project workspace; local file tools write to `sandbox/`.
The TUI asks before shell, sandbox, or subagent calls. Sessions stay in `.local/` and are
ignored by Git. Cancelling a task keeps file edits already completed.

To run generated code with `run_sandbox`, install and start
[Docker](https://docs.docker.com/get-started/get-docker/), then pull its Python image once:

```sh
docker pull python:3.12-slim
```

`run_sandbox` is a Pydantic-defined tool accepting `command` and
`timeout_seconds` (1–300, default 30). It runs in a disposable Python 3.12
container with networking disabled, a read-only system, and limits of one CPU,
256 MB RAM, and 64 processes. Only `sandbox/` is mounted at `/workspace`; changes
there persist. For example, `{"command": "python hello.py"}` runs
`sandbox/hello.py`. No host environment variables are forwarded into the container.
The host `run_bash` tool remains available and is not isolated.

For the plain terminal interface:

```sh
uv run --locked mini-harness
```
