import argparse

from mini_harness.config import CONFIG
from mini_harness.agent import DeepSeekAgent
from mini_harness.tool.box import TOOLS
from mini_harness.tool.tag import OUTCOME

EXIT = {
    OUTCOME.COMPLETED: 0,
    OUTCOME.ERROR: 1,
    OUTCOME.EXHAUSTED: 3,
    OUTCOME.TIMEOUT: 4,
    OUTCOME.INTERRUPTED: 130
}

def main(cfg = CONFIG) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default = None, help = 'the todo task')
    parser.add_argument('--telemetry-out', default = None, help = 'the json path')
    arg = parser.parse_args()

    agent = DeepSeekAgent(TOOLS, cfg = cfg)
    print(f'[mini_harness]: profile = {CONFIG.profile}, work_space = {CONFIG.work_space}, guard = {CONFIG.guard_read}/{CONFIG.guard_write}, turns = {CONFIG.max_turns_main}, wall = {cfg.wall_budget}')
    if not arg.task:
        agent.run()
        return

    result = agent.run_task(arg.task)
    agent.dump_run(result, arg.task, arg.telemetry_out)
    raise SystemExit(EXIT.get(result.outcome, 1))


if __name__ == '__main__':
    main()

    