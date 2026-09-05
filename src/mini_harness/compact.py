import time
import json

from pathlib import Path
from openai import OpenAI

from mini_harness.config import CONFIG
from mini_harness.retry_request import retry_call

class CompactContent:
    def __init__(self) -> None:
        pass

    def _get_cut(self, message: list, cfg = CONFIG) -> int:
        cut = len(message) - cfg.recent_keep
        while cut > 1 and message[cut]['role'] == 'tool':
            cut -= 1
        return cut

    def _get_user(self, old: list, cfg = CONFIG) -> str:
        hits = []
        for item in old:
            role = item.get('role')
            if role == 'user':
                hits.append(f'[user]: {item.get("content")}')
            if role == 'assistant':
                if item.get("content"):
                    hits.append(f'[assistant]: {item.get("content")}')
                if item.get("tool_calls"):
                    for tc in item.get("tool_calls"):
                        fun = tc['function']['name']
                        args = tc['function']['arguments']
                        hits.append(f'[tool call]: {fun}: {args[:200]}')
            if role == 'tool':
                hits.append(f'[tool result]: {str(item.get("content"))[:200]}')
        hits_con = '\n'.join(hits)
        return f"""
Please summarize those conversation history into a working summary report, follow the princle:
1. What is the total goal of user?
2. What items have been done? 
3. What decisions have been made?
4. What files are included in the task?
5. What step of the task right now, what is the next step
{hits_con}
"""

    def _request_agent(self, client: OpenAI, user_prompt: str, cfg = CONFIG) -> str|None:
        response = client.chat.completions.create(
            model = cfg.model_sub,
            messages = [
                {
                    'role': 'system',
                    'content': 'you are a summarizing agent whose responsibility is to summarize the history conversation by giving the summary report'
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ],
            temperature = cfg.temp_set,
            extra_body = cfg.thinking_sub,
            max_tokens = cfg.max_tokens_sub,
            stream = False
        )
        return response.choices[0].message.content

    def _compact_text(self, cut: int, response: str, message: list, cfg = CONFIG) -> list:
        return [
            message[0],
            {'role': 'assistant', 'content': f'Here is the summary of the history conversation: {response}', 'reasoning_content': ''},
            *message[cut:]
        ]

    def compact_content(self, client: OpenAI, message: list, session_path: str|Path|None = None,  cfg = CONFIG) -> list:
        cut = self._get_cut(message, cfg = cfg)
        if cut <= 1:
            return message
        old = message[1:cut]
        user_prompt = self._get_user(old, cfg = cfg)
        start = time.time()
        print(f'[compact content]: compacting the content....')
        try:
            response = retry_call(lambda: self._request_agent(client, user_prompt, cfg = cfg), cfg = cfg)
            if session_path is not None:
                try:
                    hist = Path(session_path).with_name('mini_harness_history.jsonl')
                    with open(hist, 'a', encoding = 'utf-8') as f:
                        f.write(json.dumps({'ts': time.time(), 'removed': old}, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f'[compact content]: history dump failed: {type(e).__name__}: {e}')
            message_new = self._compact_text(cut, response, message, cfg = cfg)
        except Exception as e:
            print(f'[compact content]: compact failed: {type(e).__name__}: {e}, skip this round')
            return message
        end = time.time() -start
        print(f'[compact content]: compact done -- {end:.1f}s')
        return message_new


COMPACT = CompactContent()
            
            

