"""Regression checks for vendor import and project-relative staging."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from hardware.scripts import build_fpga as build
from shared.paths import ConfigurationError


class FpgaBuildTests(unittest.TestCase):
    def test_full_build_inputs_exclude_frozen_firmware(self):
        rows = build.required_assets()
        self.assertEqual(len(rows), 746)
        self.assertTrue(all(row['target'].startswith(('vendor://fpga/', 'vendor://boot/')) for row in rows))

    def test_build_excludes_unused_legacy_and_generated_logs(self):
        targets = {row['target'] for row in build.required_assets()}
        self.assertFalse(any('/finedge_ddr/' in name for name in targets))
        self.assertNotIn('vendor://fpga/ip_patch/ip_patch.cfg', targets)
        self.assertNotIn('vendor://fpga/ip_patch/time_check.txt', targets)

    def test_vendor_paths_keep_project_layout_and_spaces(self):
        root = Path('build').resolve()
        self.assertEqual(build.replace_value('vendor://fpga/PS_AI IP/a.xml', root),
                         (root / 'project/PS_AI IP/a.xml').as_posix())
        self.assertEqual(build.replace_value('repo://hardware/rom/a.hex', root),
                         (root / 'hardware/rom/a.hex').as_posix())

    def test_project_uses_public_mailbox_and_shared_port(self):
        refs = {item.get('Path') for item in ET.parse(build.ROOT / 'hardware/project/reference.xpr').findall('.//File')}
        self.assertIn('repo://hardware/rtl/common/b1_mailbox.sv', refs)
        self.assertIn('repo://hardware/rtl/scheduler/shared_input_beat_port_v1.sv', refs)

    def test_ethernet_macro_continuations_have_valid_line_endings(self):
        for path in (build.ROOT / 'third_party/verilog-ethernet').rglob('*'):
            if path.is_file() and path.suffix in {'.v', '.tcl'}:
                with self.subTest(path=path.name):
                    self.assertNotIn(b'\r\r\n', path.read_bytes())

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

    def test_stage_preserves_plain_vendor_bytes_and_handles_json_bd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'repo'
            vendor = root / 'cache'
            for directory in ('hardware/project', 'hardware/scripts', 'third_party/verilog-ethernet'):
                (source / directory).mkdir(parents=True)
            (vendor / 'ip_patch').mkdir(parents=True)
            (vendor / 'ip_patch/ip_patch.cfg').write_text('old tool path', encoding='utf-8')
            original = b'\xef\xbb\xbf// vendor bytes\r\n'
            (vendor / 'unchanged.v').write_bytes(original)
            (vendor / 'design.bd').write_text('{"design": {}}', encoding='utf-8')
            (vendor / 'hidden.v').write_bytes(b'undeclared cache file')
            (source / 'hardware/project/reference.xpr').write_text(
                '<Project><File Path="vendor://fpga/PS_AI IP/a.xml"/>'
                '<Option Name="IPRepoPath" Val="$PPRDIR/PS_AI IP"/></Project>', encoding='utf-8')
            (source / 'hardware/scripts/build.tcl').write_text('set p {__PROJECT__}', encoding='utf-8')
            rows = [dict(target='vendor://fpga/' + name) for name in ('unchanged.v', 'design.bd', 'ip_patch/ip_patch.cfg')]
            with patch.object(build, 'ROOT', source), \
                    patch.object(build, 'resolve_path', side_effect=lambda ref: vendor / ref.removeprefix('vendor://fpga/')), \
                    patch.object(build, 'dependencies', return_value={'status': 'passed'}), \
                    patch.object(build, 'required_assets', return_value=rows), \
                    patch.object(build, 'tool_executable', return_value=root / 'tool/bin/vivado.bat'):
                project = build.stage(root / 'output')
            self.assertEqual((project.parent / 'unchanged.v').read_bytes(), original)
            self.assertFalse((project.parent / 'hidden.v').exists())
            self.assertEqual(ET.parse(project).find('.//File').get('Path'),
                             (project.parent / 'PS_AI IP/a.xml').as_posix())


if __name__ == '__main__':
    unittest.main()
