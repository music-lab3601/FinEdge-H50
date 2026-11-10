"""A venv launcher and its actual Python worker can have distinct PIDs."""
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from shared.resources import Queue


class QueueWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / 'candidate.json'
        self.queue = Queue(self.root)
        with self.queue.edit() as data:
            data['requests'].append(dict(request_id='request', state='active',
                                         candidate_manifest=str(self.manifest), child_pid=123,
                                         launch_token='local-test-launch'))
        self.env = dict(FINEDGE_QUEUE_REQUEST_ID='request', FINEDGE_QUEUE_LAUNCH_TOKEN='local-test-launch')

    def test_launcher_and_worker_are_both_recorded(self):
        with patch.dict(os.environ, self.env):
            request = self.queue.attach_child(self.manifest)
        self.assertEqual(request['child_pid'], 123)
        self.assertEqual(request['worker_pid'], os.getpid())
        self.assertEqual(self.queue._load()['requests'][0]['worker_pid'], os.getpid())

    def test_wrong_launch_or_manifest_cannot_attach(self):
        for env, manifest in [(dict(self.env, FINEDGE_QUEUE_LAUNCH_TOKEN='wrong'), self.manifest),
                              (self.env, self.root / 'different.json')]:
            with self.subTest(manifest=manifest), patch.dict(os.environ, env):
                with self.assertRaisesRegex(RuntimeError, 'Not the active'):
                    self.queue.attach_child(manifest, timeout_seconds=.02)

    def test_second_worker_cannot_replace_the_owner(self):
        with self.queue.edit() as data:
            data['requests'][0]['worker_pid'] = 456
        with patch.dict(os.environ, self.env), self.assertRaisesRegex(RuntimeError, 'different Python worker'):
            self.queue.attach_child(self.manifest)

    def test_completed_request_cannot_attach(self):
        with self.queue.edit() as data:
            data['requests'][0]['state'] = 'completed'
        with patch.dict(os.environ, self.env), self.assertRaisesRegex(RuntimeError, 'Not the active'):
            self.queue.attach_child(self.manifest, timeout_seconds=.02)

    def test_actual_python_launch_attaches_before_returning(self):
        queue = Queue(self.root / 'actual_launch')
        self.manifest.write_text('{}', encoding='utf-8')
        script = ('from pathlib import Path; from shared.resources import Queue; '
                  + 'Queue(Path(' + repr(str(queue.root)) + ')).attach_child(Path('
                  + repr(str(self.manifest)) + '))')
        request = queue.submit(resource='rtl_simulation', stage='small_benchmark',
                               candidate_manifest=self.manifest, candidate_sha256='test-fixture',
                               command=[sys.executable, '-B', '-c', script])
        result = queue.run(request['request_id'], qualify=lambda request: None, timeout_seconds=15)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['return_code'], 0)
        stored = queue._load()['requests'][0]
        self.assertGreater(stored['child_pid'], 0)
        self.assertGreater(stored['worker_pid'], 0)


if __name__ == '__main__':
    unittest.main()
