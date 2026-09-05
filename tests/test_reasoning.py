import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

from mini_harness.agent import DeepSeekAgent
from mini_harness.compact import COMPACT
from mini_harness.config import CONFIG
from mini_harness.tool.box import TOOLS, RunSubAgentInput, ToolExecution, ToolItem, run_subagent


class ReasoningHistoryTests(unittest.TestCase):
    def test_sdk_stream_tool_and_next_user_turn_retain_all_reasoning(self):
        requests = []
        def serve(request):
            body = json.loads(request.content)
            requests.append(body)
            number = len(requests)
            if number > 1:
                assistants = [m for m in body['messages'] if m['role'] == 'assistant']
                self.assertEqual(assistants[0]['reasoning_content'], 'First thought.')
                if number == 3:
                    self.assertEqual(assistants[1]['reasoning_content'], 'Next thought.')
            deltas = [{'role': 'assistant', 'reasoning_content': 'First ' if number == 1 else 'Next '},
                      {'reasoning_content': 'thought.'}]
            if number == 1:
                deltas += [{'tool_calls': [{'index': 0, 'id': 'call_1', 'type': 'function',
                            'function': {'name': 'run_todo', 'arguments': json.dumps({'items': [
                                {'content': 'Demo', 'activeForm': 'Demonstrating', 'status': 'completed'}]})}}]}]
            else:
                deltas += [{'content': 'Done.'}]
            chunks = [{'id': 'test', 'object': 'chat.completion.chunk', 'created': 0,
                       'model': 'deepseek-v4-flash', 'choices': [{'index': 0, 'delta': delta,
                       'finish_reason': None}]} for delta in deltas]
            chunks.append({**chunks[-1], 'choices': [{'index': 0, 'delta': {},
                          'finish_reason': 'tool_calls' if number == 1 else 'stop'}],
                          'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}})
            sse = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks) + 'data: [DONE]\n\n'
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=sse)

        with tempfile.TemporaryDirectory() as tmp, OpenAI(api_key='test', base_url='https://test.invalid',
                http_client=httpx.Client(transport=httpx.MockTransport(serve)), max_retries=0) as client:
            cfg = replace(CONFIG, session_path=Path(tmp) / 'session.json', max_retry=0)
            agent = DeepSeekAgent(TOOLS, cfg)
            executor = ToolExecution(agent.regis, lambda call: True)
            with contextlib.redirect_stdout(io.StringIO()):
                agent.message.append({'role': 'user', 'content': 'Make a todo.'})
                self.assertFalse(agent._run_turn(client, executor, cfg).err)
                agent.message.append({'role': 'user', 'content': 'What next?'})
                self.assertFalse(agent._run_turn(client, executor, cfg).err)
            self.assertEqual(len(requests), 3)
            saved = json.loads(Path(agent.session_memory).read_text())
            self.assertEqual(saved[-1]['reasoning_content'], 'Next thought.')

    def test_nonstreaming_subagent_keeps_reasoning(self):
        requests = []
        def serve(request):
            body = json.loads(request.content)
            requests.append(body)
            message = {'role': 'assistant', 'content': 'Finished', 'reasoning_content': 'Keep me.'}
            if len(requests) == 1:
                message['tool_calls'] = [{'id': 'call_sub', 'type': 'function', 'function':
                                         {'name': 'read_file', 'arguments': '{"file_path":"demo.py"}'}}]
            else:
                self.assertEqual(body['messages'][-2]['reasoning_content'], 'Keep me.')
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'deepseek-v4-flash', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}]})
        with OpenAI(api_key='test', base_url='https://test.invalid',
                    http_client=httpx.Client(transport=httpx.MockTransport(serve))) as client:
            with patch('mini_harness.tool.box.OpenAI', return_value=client), \
                 patch.object(ToolExecution, 'execute_tool', return_value=ToolItem('read', True, 'ok')), \
                 contextlib.redirect_stdout(io.StringIO()):
                answer = run_subagent(RunSubAgentInput(task_description='test', prompt='read demo.py',
                                                       agent_type='explore_agent'))
        self.assertEqual(answer, 'Finished')
        self.assertEqual(len(requests), 2)

    def test_compaction_preserves_recent_reasoning(self):
        recent = {'role': 'assistant', 'content': 'Recent answer', 'reasoning_content': 'Recent thought'}
        history = COMPACT._compact_text(2, 'summary', [{'role': 'system', 'content': 'system'},
                                                      {'role': 'user', 'content': 'old'}, recent])
        self.assertEqual(history[1]['reasoning_content'], '')
        self.assertEqual(history[-1], recent)


if __name__ == '__main__':
    unittest.main()
