"""Regression checks for the relocated public entry points. No board access."""
import http.cookiejar
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request

from shared import paths
from tools.configuration import render_templates


class ConfigurationTests(unittest.TestCase):
    def test_uri_roots_and_traversal(self):
        self.assertEqual(paths.resolve_path('repo://models/npu/model.json'), paths.ROOT/'models/npu/model.json')
        for ref in ('repo://../x', 'data://../x', 'vendor://a/../../x', 'work://a\\b'):
            with self.subTest(reference=ref), self.assertRaises(paths.ConfigurationError):
                paths.resolve_path(ref)

        import xml.etree.ElementTree as ET
        external=json.loads((paths.ROOT/'external/dependencies.json').read_text('utf-8'))
        declared={row['target'] for row in external['files']}
        for step in ET.parse(paths.ROOT/'hardware/project/reference.xpr').findall('.//Step'):
            for name in ('PreStepTclHook','PostStepTclHook'):
                reference=step.get(name)
                if reference:
                    self.assertTrue(reference in declared or (reference.startswith('repo://') and paths.resolve_path(reference).is_file()))

    def test_defaults_follow_relocated_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            other = Path(temporary)
            (other/'config').mkdir()
            (other/'config/default.json').write_bytes(paths.DEFAULT_CONFIG.read_bytes())
            with patch.object(paths, 'ROOT', other), patch.object(paths, 'DEFAULT_CONFIG', other/'config/default.json'), patch.dict('os.environ', {'FINEDGE_CONFIG': str(other/'absent.json')}):
                # Explicit overrides are optional when this test uses no local file.
                self.assertEqual(paths.resolve_path('work://runs'), other/'.local/runs')
                self.assertEqual(paths.resolve_path('data://test.bin'), other/'.local/data/test.bin')

    def test_render_all_original_templates_is_repeatable(self):
        first = render_templates()
        target = paths.work_path('rendered')
        before = {row['path']: (target/row['path']).read_bytes() for row in first['files']}
        second = render_templates()
        self.assertEqual(first['rendered_templates'], 57)
        self.assertEqual(first, second)
        self.assertTrue(all((target/name).read_bytes() == content for name, content in before.items()))

    def test_native_scales_match_frozen_model(self):
        import numpy as np
        from deployment.icraft.native.codec3 import SCALE, encode, decode
        self.assertEqual(SCALE.tobytes(), (paths.ROOT/'models/npu/scales.f32').read_bytes())
        sample = np.zeros((1,40,100), dtype='<f4')
        self.assertEqual(len(encode(sample)), 4800)
        np.testing.assert_array_equal(decode(encode(sample)), sample)


class DashboardTests(unittest.TestCase):
    def test_idle_dashboard_static_and_authenticated_api(self):
        from host.dashboard.app import App, make_server
        app = App()
        server = make_server(app)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            address = 'http://127.0.0.1:' + str(server.server_port)
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            self.assertIn(b'html', opener.open(address,timeout=3).read().lower())
            for path in ('/app.js', '/app.css', '/vendor/lightweight-charts.js'):
                self.assertGreater(len(opener.open(address+path,timeout=3).read()), 100)
            state = json.load(opener.open(address+'/api/state',timeout=3))
            self.assertEqual(state['phase'], 'idle')
            self.assertFalse(state['CPU_reference'])
            self.assertIsNone(app.worker)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_missing_board_inputs_fail_before_console_access(self):
        from shared.config import board_config
        from tools.board_bundle import preflight
        cfg = board_config()
        cfg['raw_source'] = str(paths.work_path('missing/input.bin'))
        with patch('board.management.serial_console.Console', side_effect=AssertionError('Board accessed')):
            with self.assertRaises(paths.ConfigurationError):
                preflight(cfg)


if __name__ == '__main__':
    unittest.main()
