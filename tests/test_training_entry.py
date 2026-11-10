"""Real CLI checks for portable H50 cache admission and one synthetic epoch."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from shared.io import sha
from shared.paths import ROOT
from software.training import entry
from tests.software.test_tcn_existing_data_ed4_features import fixture


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    return sha(path)


class TrainingEntryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.data, self.cache = fixture(self.root)
        self.data['normalization'] = copy.deepcopy(self.cache['normalization'])
        self.data['data_kind'] = 'synthetic'
        self.data_path = self.root/'data.json'
        self.data_sha = save(self.data_path, self.data)
        self.cache['p2_manifest_binding'] = dict(path=str(self.data_path), sha256=self.data_sha)
        self.cache_path = self.root/'cache.json'
        self.cache_sha = save(self.cache_path, self.cache)

    def tearDown(self):
        self.temporary.cleanup()

    def test_real_cli_prepare_admit_and_one_epoch(self):
        prepared = self.root/'prepared'
        command = [sys.executable, '-B', str(ROOT/'finedge.py'), 'train']
        common = ['--data-manifest', str(self.data_path), '--data-sha256', self.data_sha]
        process = subprocess.run(command+['prepare', *common, '--output', str(prepared)], cwd=ROOT,
                                 capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        cache_path = prepared/'cache_manifest.json'
        common += ['--cache-manifest', str(cache_path), '--cache-sha256', sha(cache_path)]
        process = subprocess.run(command+['admit', *common], cwd=ROOT,
                                 capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        self.assertEqual(json.loads(process.stdout)['sample_counts'], {'train': 30, 'tuning_validation': 12})
        output = self.root/'run'
        process = subprocess.run(command+['run', *common, '--output', str(output), '--epochs', '1', '--device', 'cpu'],
                                 cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        result = json.loads((output/'result.json').read_text('utf-8'))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['data_kind'], 'synthetic')
        self.assertEqual(result['optimizer_updates'], 1)
        self.assertEqual(result['history'][0]['train']['samples'], 30)
        self.assertEqual(result['history'][0]['development']['samples'], 12)
        import torch
        from software.model.tcn_research_p5_model import build_model
        checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
        model = build_model(weight_norm=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        self.assertEqual(entry.state_digest(model), checkpoint['state_sha256'])
        self.assertNotEqual(result['initial_state_sha256'], checkpoint['state_sha256'])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 33539)

    def test_changed_metadata_and_label_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Metadata hash'):
            entry.inputs(self.data_path, '0'*64, self.cache_path, self.cache_sha)
        label = Path(self.data['day_label_reports']['2025-12-08']['label_path'])
        label.write_bytes(b'\x00'*6)
        with self.assertRaisesRegex(ValueError, 'Label stream changed'):
            entry.inputs(self.data_path, self.data_sha, self.cache_path, self.cache_sha)

    def test_cache_bound_to_explicit_parent(self):
        self.cache['p2_manifest_binding']['sha256'] = '0'*64
        digest = save(self.cache_path, self.cache)
        with self.assertRaisesRegex(ValueError, 'parent manifest differs'):
            entry.inputs(self.data_path, self.data_sha, self.cache_path, digest)

    def test_relative_metadata_can_be_relocated(self):
        data = copy.deepcopy(self.data)
        for source in data['sources'].values():
            source['records_path'] = Path(source['records_path']).relative_to(self.root).as_posix()
        for row in data['ranges']:
            row['records_path'] = Path(row['records_path']).relative_to(self.root).as_posix()
        for report in data['day_label_reports'].values():
            report['label_path'] = Path(report['label_path']).relative_to(self.root).as_posix()
        digest = save(self.data_path, data)
        loaded = entry.metadata(self.data_path, digest)
        self.assertEqual(loaded['sources'], self.data['sources'])
        self.assertEqual(loaded['ranges'], self.data['ranges'])

    def test_existing_output_is_preserved(self):
        output = self.root/'existing'
        output.mkdir()
        sentinel = output/'keep.txt'
        sentinel.write_text('keep', encoding='utf-8')
        data, cache, roots = entry.inputs(self.data_path, self.data_sha, self.cache_path, self.cache_sha)
        with self.assertRaises(FileExistsError):
            entry.train(data, cache, roots, output, 'cpu', 1, {})
        self.assertEqual(sentinel.read_text('utf-8'), 'keep')

    def test_historical_registration_argument_forms_are_forwarded(self):
        for arguments in (['--registration', 'input.json'], ['--registration=input.json']):
            with self.subTest(arguments=arguments), patch(
                    'software.training.tcn_existing_data_ed4_train.main', return_value=7) as historical:
                self.assertEqual(entry.main(arguments), 7)
                historical.assert_called_once_with(arguments)


if __name__ == '__main__':
    unittest.main()
