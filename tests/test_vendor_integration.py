"""Check public integration recipes without distributing proprietary base files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hardware.scripts import integrate_vendor as integration
from shared.paths import ConfigurationError, ROOT


def recipe():
    before = b'licensed prefix\r\nold application\r\nlicensed suffix\r\n'
    after = before.replace(b'old application', b'our application')
    start = before.index(b'old application')
    return before, after, dict(target='vendor://fpga/rtl/top.v', base_package='fpai_demo_fpga',
        base_path='rtl/top.v', base_bytes=len(before), base_sha256=hashlib.sha256(before).hexdigest(),
        output_bytes=len(after), output_sha256=hashlib.sha256(after).hexdigest(),
        edits=[dict(start=start, end=start+len(b'old application'), replacement='our application')])


class IntegrationTests(unittest.TestCase):
    def test_exact_reconstruction_and_repeat(self):
        before, after, row = recipe()
        self.assertEqual(integration.reconstruct(row, before), after)
        self.assertEqual(integration.reconstruct(row, after), after)

    def test_line_endings_are_converted_before_byte_edits(self):
        before, after, row = recipe()
        original = before.replace(b'\r\n', b'\n')
        row.update(base_bytes=len(original), base_sha256=hashlib.sha256(original).hexdigest(), line_endings='crlf')
        self.assertEqual(integration.reconstruct(row, original), after)

    def test_reject_changed_base_and_result(self):
        before, _, row = recipe()
        with self.assertRaises(ConfigurationError):
            integration.reconstruct(row, before + b'changed')
        row['edits'][0]['replacement'] = 'broken application'
        with self.assertRaises(ConfigurationError):
            integration.reconstruct(row, before)

    def test_reject_invalid_edit_ranges(self):
        before, _, row = recipe()
        for start, end in ((-1, 0), (10, 2), (0, len(before)+1)):
            with self.subTest(start=start, end=end):
                row['edits'][0].update(start=start, end=end)
                with self.assertRaises(ConfigurationError):
                    integration.reconstruct(row, before)

    def test_validates_all_destinations_before_writing(self):
        before, after, row = recipe()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base/'original/rtl/top.v'
            source.parent.mkdir(parents=True)
            source.write_bytes(before)
            target = base/'cache/rtl/top.v'
            with patch.object(integration, 'recipes', return_value=[row]), \
                    patch.object(integration, 'resolve_path', side_effect=lambda ref: base/'cache'/ref.removeprefix('vendor://').removeprefix('fpga/')):
                result = integration.integrate(base/'original', base/'original')
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(target.read_bytes(), after)
                integration.integrate(base/'original', base/'original')
                target.write_bytes(b'keep existing file')
                with self.assertRaises(ConfigurationError):
                    integration.integrate(base/'original', base/'original')
                self.assertEqual(target.read_bytes(), b'keep existing file')

    def test_no_partial_writes_for_late_base_failure(self):
        before, _, row = recipe()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base/'original/rtl/top.v'
            source.parent.mkdir(parents=True)
            source.write_bytes(before)
            (source.parent/'other.v').write_bytes(b'wrong base')
            second = dict(row, base_path='rtl/other.v', target='vendor://fpga/rtl/other.v')
            with patch.object(integration, 'recipes', return_value=[row, second]), \
                    patch.object(integration, 'resolve_path', side_effect=lambda ref: base/'cache'/ref.removeprefix('vendor://').removeprefix('fpga/')):
                with self.assertRaises(ConfigurationError):
                    integration.integrate(base/'original', base/'original')
            self.assertFalse((base/'cache').exists())

    def test_recipe_covers_top_peripheral_constraints_and_hooks(self):
        entries = integration.recipes()
        targets = {row['target'] for row in entries}
        self.assertEqual(len(entries), 8)
        self.assertIn('vendor://fpga/rtl/ai7030_edif_top.v', targets)
        self.assertIn('vendor://fpga/rtl/ps_ai_peripheral.sv', targets)
        self.assertIn('vendor://fpga/constrs_1/impl_constraints.xdc', targets)
        manifest = {row['target']: row for row in json.loads((ROOT/'external/dependencies.json').read_text('utf-8'))['files']}
        for row in entries:
            self.assertEqual(row['output_sha256'], manifest[row['target']]['sha256'])
            self.assertEqual(row['output_bytes'], manifest[row['target']]['bytes'])
            self.assertTrue(manifest[row['target']]['exact_reconstruction_verified'])
        for row in entries:
            for edit in row['edits']:
                self.assertNotIn('Confidential', edit['replacement'])


if __name__ == '__main__':
    unittest.main()
