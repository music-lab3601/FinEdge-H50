"""CI must preserve diagnostics and propagate a failed command."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

from tools.ci_check import run_step


class CICheckTests(unittest.TestCase):
    def test_success_is_recorded_with_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'logs').mkdir()
            steps = []
            run_step('ok', [sys.executable, '-c', 'print("ci smoke")'], output, dict(os.environ), steps)
            self.assertEqual(steps[0]['status'], 'passed')
            self.assertEqual(steps[0]['return_code'], 0)
            self.assertIn('ci smoke', (output / 'logs/ok.log').read_text('utf-8'))

    def test_failure_is_recorded_and_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'logs').mkdir()
            steps = []
            command = [sys.executable, '-c', 'import sys; print("fault injection"); sys.exit(7)']
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                run_step('bad', command, output, dict(os.environ), steps)
            self.assertEqual(steps[0]['status'], 'failed')
            self.assertEqual(steps[0]['return_code'], 7)
            self.assertIn('fault injection', (output / 'logs/bad.log').read_text('utf-8'))


if __name__ == '__main__':
    unittest.main()
