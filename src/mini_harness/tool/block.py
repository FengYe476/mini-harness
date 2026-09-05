from mini_harness.config import CONFIG

class TodoManager:
    def __init__(self) -> None:
        self.items = []
        return

    def update(self, items: list) -> str:
        self.items = items
        return self.render()

    def render(self) -> str:
        if not self.items:
            return 'no items'
        action = []
        for item in self.items:
            if item.status == 'completed':
                action.append(f'[x] {item.content}')
            elif item.status == 'in_progress':
                action.append(f'[>] {item.activeForm}')
            else:
                action.append(f'[ ] {item.content}')
        completed = sum(1 for t in self.items if t.status == 'completed')
        action.append(f'{completed} / {len(self.items)} completed')
        return '\n'.join(action)

class Clip:
    def __init__(self) -> None:
        pass

    def clip(self, content: str, cfg = CONFIG) -> str:
        if len(content) < cfg.clip_limit:
            return content
        return content[:cfg.clip_limit] + f'\n.....clipped at {cfg.clip_limit} of {len(content)} chars'
    
class SubAgent:
    def __init__(self) -> None:
        self.agent_table = {
            'explore_agent': {
                'description': 'explore, read and find the file and file content based on the task',
                'tools': ['read_file', 'glob_file', 'grep_file'],
                'prompt':  """
You are a explore-agent, your job is to read and analyze the file. Here is your chain of thought:
step1: scan the file: Use the tool "read_file", "grep_file" to scan and check the file content which you are interested
step2: Analyze the file: try to analyze the situation of the task and read the whole content which you want to read by using the tool "read_file"
"""
            },
            'coding_agent': {
                'description': 'create, code and program the file based on the task',
                'tools': ['read_file', 'glob_file', 'grep_file', 'write_file', 'edit_file'],
                'prompt': """
You are a coding-agent, your job is to code and write the file. Here is you chain of thought:
step1: thinking about the task(question): Use the tool "read_file", "glob_file", "grep_file" to know about the situation real_time
step2: design multiple way of solution: try to thinking about some strategy which can solve the question
step3: pick the besk solution and start writing: pick the best solution to write the content, use the tool like, "write_file", "edit_file" to write the content
"""
            },
            'planning_agent': {
                'description': 'read, analyze and make a plan to better finish the task',
                'tools': ['read_file', 'glob_file', 'grep_file', 'write_file', 'edit_file'],
                'prompt': """
You are a planing_agent, your jon is to make plan of the task. Here is your chain of thought:
step1: thinking about the taks: try to analyzing the goal of user, and use the tool "read_file", 'glob_file", "grep_file" to know about the situation,
step2: write the plan: after analyzing, use the tool "write_file", "edit_file" to write a md file to show the plan and give the user your analzing report and the path of md file
"""
            }
            
        }

TODO = TodoManager()
CLIP = Clip()
SUBAGENT = SubAgent()
