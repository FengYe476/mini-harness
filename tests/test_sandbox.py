import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

from mini_harness.config import CONFIG
from mini_harness.tool.box import TOOLS, RunSandboxInput, ToolExecution, run_sandbox


class SandboxContractTests(unittest.TestCase):
    def test_invalid_arguments_and_denial_never_execute(self):
        tool = next(t for t in TOOLS if t.name == 'run_sandbox')
        function = Mock()
        executor = ToolExecution({tool.name: replace(tool, function=function)}, lambda call, **kwargs: False)
        for args in ({'command': ' '}, {'command': 'pwd', 'timeout_seconds': 301},
                     {'command': 'pwd', 'timeout_seconds': True},
                     {'command': 'pwd', 'image': 'untrusted'}, {'command': 'pwd'}):
            call = SimpleNamespace(function=SimpleNamespace(name=tool.name, arguments=json.dumps(args)))
            self.assertFalse(executor.execute_tool(call).ok)
        function.assert_not_called()
        with self.assertRaises(ValidationError):
            RunSandboxInput(command='pwd', timeout_seconds='30')

    def test_symlinked_workspace_rejected(self):
        with tempfile.TemporaryDirectory() as name, tempfile.TemporaryDirectory() as outside:
            root = Path(name)
            (root / 'sandbox').symlink_to(outside, target_is_directory=True)
            with patch('mini_harness.tool.box.shutil.which', return_value='/usr/bin/docker'):
                with self.assertRaises(PermissionError):
                    run_sandbox(RunSandboxInput(command='pwd'), replace(CONFIG, work_space=root))

    def test_missing_docker_is_actionable(self):
        with patch('mini_harness.tool.box.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'requires Docker'):
                run_sandbox(RunSandboxInput(command='pwd'))


@unittest.skipUnless(os.environ.get('MINI_HARNESS_TEST_DOCKER') == '1', 'set MINI_HARNESS_TEST_DOCKER=1 for Docker integration')
class DockerSandboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mini-harness-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cfg = replace(CONFIG, work_space=self.root, bash_limit=512)
        subprocess.run(['docker', 'image', 'inspect', 'python:3.12-slim'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.before = self.containers()

    def containers(self):
        return set(subprocess.check_output(['docker', 'ps', '-aq', '--filter',
                                           'label=mini-harness.sandbox=true'], text=True).split())

    def tearDown(self):
        self.assertEqual(self.containers(), self.before, 'sandbox container leaked')

    def test_real_isolation_and_persistent_file(self):
        outside = self.root / 'host-only.txt'
        outside.write_text('host marker')
        code = f'''
import os, socket
from pathlib import Path
assert not Path({str(outside)!r}).exists()
assert 'DEEPSEEK_API_KEY' not in os.environ
assert 'SANDBOX_TEST_SECRET' not in os.environ
assert not Path('/var/run/docker.sock').exists()
assert Path('/proc/sys/kernel/hostname').read_text().strip() != ''
try:
    Path('/etc/sandbox-test').write_text('blocked')
except OSError:
    pass
else:
    raise AssertionError('root filesystem is writable')
try:
    socket.create_connection(('1.1.1.1', 443), timeout=1)
except OSError:
    pass
else:
    raise AssertionError('network is reachable')
Path('result.txt').write_text('sandbox wrote this')
print('ISOLATION OK')
'''
        with patch.dict(os.environ, {'SANDBOX_TEST_SECRET': 'do-not-forward'}):
            result = run_sandbox(RunSandboxInput(command='python -c ' + shlex.quote(code)), self.cfg)
        self.assertIn('ISOLATION OK', result)
        self.assertEqual((self.root / 'sandbox/result.txt').read_text(), 'sandbox wrote this')
        self.assertEqual(outside.read_text(), 'host marker')

    def test_timeout_cleanup(self):
        with self.assertRaisesRegex(TimeoutError, 'exceeded 1s'):
            run_sandbox(RunSandboxInput(command='sleep 20', timeout_seconds=1), self.cfg)

    def test_failure_and_bounded_output(self):
        with self.assertRaisesRegex(RuntimeError, 'code 7'):
            run_sandbox(RunSandboxInput(command='echo failed; exit 7'), self.cfg)
        text = run_sandbox(RunSandboxInput(command="python -c 'print(\"x\" * 1000000)'"), self.cfg)
        self.assertIn('truncated at 512 bytes', text)
        self.assertLess(len(text), 600)

    def test_interrupt_removes_container(self):
        code = '''
from mini_harness.tool.box import run_sandbox, RunSandboxInput
run_sandbox(RunSandboxInput(command='sleep 60', timeout_seconds=60))
'''
        process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True,
                                   env={**os.environ, 'MINI_HARNESS_WORK_SPACE': str(self.root)})
        try:
            deadline = time.monotonic() + 15
            while self.containers() == self.before:
                self.assertLess(time.monotonic(), deadline, 'container never started')
                time.sleep(0.1)
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=6)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == '__main__':
    unittest.main()
