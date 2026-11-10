"""Regression checks for missing files outside the fixed key-file list."""
from pathlib import Path
import tempfile
import unittest

from tools.source_closure import check


class SourceClosureTests(unittest.TestCase):
    def test_missing_document_and_untracked_native_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'host').mkdir()
            (root/'README.md').write_text('[build](docs/build.md)', encoding='utf-8')
            (root/'host/app.cpp').write_text('#include "app.h"', encoding='utf-8')
            (root/'host/app.h').write_text('// header', encoding='utf-8')
            result = check(root, ['README.md', 'host/app.cpp'])
            self.assertEqual(len(result['errors']), 2)
            self.assertIn('absent from Git', result['errors'][1])

    def test_hdl_include_and_rom_and_python_resource(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'hardware/rtl').mkdir(parents=True)
            (root/'hardware/rom').mkdir()
            (root/'hardware/rtl/a.sv').write_text('`include "params.vh"\nparameter F="repo://hardware/rom/data.hex";', encoding='utf-8')
            (root/'hardware/rom/params.vh').write_text('// params', encoding='utf-8')
            (root/'hardware/rom/data.hex').write_text('00', encoding='utf-8')
            (root/'host').mkdir()
            (root/'host/app.py').write_text('MODEL="repo://missing.model"', encoding='utf-8')
            files = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()]
            result = check(root, files)
            self.assertEqual(len(result['errors']), 1)
            self.assertIn('missing.model', result['errors'][0])


if __name__ == '__main__':
    unittest.main()
