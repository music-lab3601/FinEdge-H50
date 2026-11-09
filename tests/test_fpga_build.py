"""Regression checks for vendor import and project-relative staging."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hardware.scripts import build_fpga as build
from shared.paths import ConfigurationError


class FpgaBuildTests(unittest.TestCase):
    def test_full_build_inputs_exclude_frozen_firmware(self):
        rows = build.required_assets()
        self.assertEqual(len(rows), 756)
        self.assertTrue(all(row['target'].startswith(('vendor://fpga/', 'vendor://boot/')) for row in rows))

    def test_vendor_paths_keep_project_layout_and_spaces(self):
        root = Path('build').resolve()
        self.assertEqual(build.replace_value('vendor://fpga/PS_AI IP/a.xml', root),
                         (root / 'project/PS_AI IP/a.xml').as_posix())
        self.assertEqual(build.replace_value('repo://hardware/rom/a.hex', root),
                         (root / 'hardware/rom/a.hex').as_posix())

    def test_import_rejects_changed_source_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'input.bin').write_bytes(b'changed')
            rows = [dict(target='vendor://fpga/input.bin', original_target='hardware/input.bin',
                         bytes=4, sha256=hashlib.sha256(b'good').hexdigest())]
            with patch.object(build, 'required_assets', return_value=rows):
                with self.assertRaises(ConfigurationError):
                    build.import_assets(root)

    def test_staging_rejects_spaces_before_accessing_vendor(self):
        with self.assertRaises(ConfigurationError):
            build.stage(Path('build with spaces'))

    def test_restore_deleted_checkpoint_checks_manifest_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'vendor.dcp'
            cache.write_bytes(b'IP-checkpoint')
            (root / 'staging.json').write_text('{}', encoding='utf-8')
            row = dict(target='vendor://fpga/rtl/ip/a.dcp', sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
            with patch.object(build, 'required_assets', return_value=[row]), patch.object(build, 'resolve_path', return_value=cache):
                result = build.restore_vendor_outputs(root)
                self.assertEqual(result['restored'], [row['target']])
                self.assertEqual(result['verified_IP_checkpoints'], [row['target']])
                (root / 'project/rtl/ip/a.dcp').write_bytes(b'changed')
                with self.assertRaises(ConfigurationError):
                    build.restore_vendor_outputs(root)


if __name__ == '__main__':
    unittest.main()
