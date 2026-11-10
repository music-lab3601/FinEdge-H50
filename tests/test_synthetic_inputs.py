"""Public input/profile regressions. All board access stays outside these tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shared.config import board_config
from shared.io import sha
from shared.paths import ConfigurationError
from tools import synthetic_inputs as sample
from tools.board_bundle import prepare_integrated_bundle


class SyntheticInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / 'inputs'
        sample.generate(self.inputs)
        self.profile = self.inputs / 'manifest.json'

    def test_generation_is_byte_stable_and_has_H50_tail(self):
        other = self.root / 'other'
        result = sample.generate(other)
        self.assertEqual(result['events'], 3070)
        self.assertEqual(result['rows'], 3050)
        for path in self.inputs.rglob('*'):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (other / path.relative_to(self.inputs)).read_bytes())
        manifest, files = sample.load_profile(self.profile)
        self.assertEqual(manifest['data_kind'], 'synthetic')
        ranges = json.loads(files['duration_data'].read_text('utf-8'))
        for minutes, count in ((1, 600), (2, 1200), (5, 3000)):
            self.assertEqual(ranges['ranges'][str(minutes)]['opportunities'], count - 99)
            self.assertGreaterEqual(sample.ROWS, count + 50)

    def test_modified_raw_rejected_even_with_rehashed_manifest(self):
        manifest = json.loads(self.profile.read_text('utf-8'))
        path = self.inputs / sample.FILES['raw_source']
        blob = bytearray(path.read_bytes())
        blob[-1] ^= 1
        path.write_bytes(blob)
        manifest['files']['raw_source']['sha256'] = sha(path)
        self.profile.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(ConfigurationError):
            sample.verify(self.profile)

    def test_model_or_relative_path_changes_are_rejected(self):
        manifest = json.loads(self.profile.read_text('utf-8'))
        manifest['files']['raw_source']['path'] = '../outside.bin'
        self.profile.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(ConfigurationError):
            sample.load_profile(self.profile)
        manifest['files']['raw_source']['path'] = sample.FILES['raw_source']
        manifest['model_identity']['model'] = '0' * 64
        self.profile.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(ConfigurationError):
            sample.load_profile(self.profile)

    def test_profile_connects_board_fixture_and_standalone_bundle(self):
        with patch.dict('os.environ', {'FINEDGE_CONFIG': str(self.root / 'absent.json')}):
            config = board_config(profile=self.profile)
            self.assertEqual(config['dataset']['symbol'], 'SYNTH')
            self.assertEqual(config['raw_source_sha256'], sha(self.inputs / sample.FILES['raw_source']))
            from host.pipeline import prepare_fixture
            for minutes in (1, 2, 5):
                directory = self.root / f'range_{minutes}'
                directory.mkdir()
                _, fixture = prepare_fixture(directory, minutes, config)
                self.assertEqual(fixture['eligible_rows'], minutes * 600)
            bundle = self.root / 'bundle'
            result = prepare_integrated_bundle(bundle, config)
        self.assertEqual(len(result['files']), 11)
        self.assertEqual(result['dataset']['data_kind'], 'synthetic')
        self.assertTrue((bundle / 'integrated_owner.py').is_file())
        self.assertTrue((bundle / 'startup_input.py').is_file())
        self.assertEqual((bundle / 'dual_slot_input.i8').stat().st_size, 9600)
        for name, digest in result['files'].items():
            self.assertEqual(sha(bundle / name), digest)

    def test_startup_change_rejected_before_bundle(self):
        path = self.inputs / sample.FILES['golden_logits']
        path.write_bytes(bytes(12))
        with self.assertRaises(ConfigurationError):
            board_config(profile=self.profile)

    def test_dashboard_labels_synthetic_data_and_rejects_wrong_review_reference(self):
        local = self.root / 'config.json'
        local.write_text(json.dumps(dict(paths=dict(work_root=str(self.root / 'work')),
            inputs=dict(demo_profile=str(self.profile)))), encoding='utf-8')
        with patch.dict('os.environ', {'FINEDGE_CONFIG': str(local)}):
            from host.dashboard.app import App
            app = App()
            self.assertEqual(app.state()['dataset']['symbol'], 'SYNTH')
            review = self.root / 'wrong_review'
            review.mkdir()
            (review / 'run_manifest.json').write_text(json.dumps(dict(reference_sha256='0' * 64)), encoding='utf-8')
            with self.assertRaises(ValueError):
                App(review)

    def test_explicit_firmware_and_rollback_identities_are_resolved(self):
        candidate = self.root / 'candidate.json'
        previous = self.root / 'previous.json'
        candidate.write_text(json.dumps(dict(boot_sha256='a' * 64)), encoding='utf-8')
        previous.write_text(json.dumps(dict(boot_sha256='b' * 64)), encoding='utf-8')
        local = self.root / 'config.json'
        local.write_text(json.dumps(dict(inputs=dict(firmware_release=str(candidate),
            rollback_firmware_release=str(previous)))), encoding='utf-8')
        with patch.dict('os.environ', {'FINEDGE_CONFIG': str(local)}):
            config = board_config()
        self.assertEqual(config['BOOT_sha256'], 'a' * 64)
        self.assertEqual(config['rollback_BOOT_sha256'], 'b' * 64)
        self.assertEqual(config['candidate_manifest'], str(candidate))

    def test_cpu_receives_and_reconstructs_synthetic_UDP_events(self):
        result = sample.cpu_replay(self.profile, self.root / 'cpu', records=100, speed=1000)
        self.assertEqual(result['records'], 100)
        self.assertEqual(result['windows'], 1)
        self.assertEqual(result['backend'], 'CPU')
        self.assertFalse(result['board_access'])
        self.assertFalse(result['market_accuracy_evaluated'])


if __name__ == '__main__':
    unittest.main()
