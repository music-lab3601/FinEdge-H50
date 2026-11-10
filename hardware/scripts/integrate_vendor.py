"""Reconstruct the validated FPGA integration from licensed original files."""
import hashlib
import json
from pathlib import Path

from shared.io import atomic_json
from shared.paths import ConfigurationError, ROOT, resolve_path


RECIPE = ROOT / 'hardware/integration/vendor_edits.json'


def recipes():
    return json.loads(RECIPE.read_text('utf-8'))['files']


def reconstruct(row, original):
    """Apply ordered byte edits, accepting an already-integrated input as well."""
    digest = hashlib.sha256(original).hexdigest()
    if digest == row['output_sha256'] and len(original) == row['output_bytes']:
        return original
    if digest != row['base_sha256'] or len(original) != row['base_bytes']:
        raise ConfigurationError('Integration base identity differs: ' + row['target'])
    if row.get('line_endings', 'preserve') == 'crlf':
        original = original.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    parts = []
    cursor = 0
    for edit in row['edits']:
        start, end = edit['start'], edit['end']
        if not cursor <= start <= end <= len(original):
            raise ConfigurationError('Invalid integration edit range: ' + row['target'])
        parts.extend((original[cursor:start], edit['replacement'].encode('utf-8')))
        cursor = end
    result = b''.join(parts) + original[cursor:]
    if len(result) != row['output_bytes'] or hashlib.sha256(result).hexdigest() != row['output_sha256']:
        raise ConfigurationError('Integration result identity differs: ' + row['target'])
    return result


def integrate(reference, jfm):
    """Validate every base and destination before creating any integration file."""
    bases = {'fpai_demo_fpga': reference.resolve(strict=True), 'JFM_Kits': jfm.resolve(strict=True)}
    pending = []
    for row in recipes():
        base = bases[row['base_package']]
        source = (base / row['base_path']).resolve(strict=True)
        if not source.is_relative_to(base):
            raise ConfigurationError('Integration base escapes package')
        data = reconstruct(row, source.read_bytes())
        target = resolve_path(row['target'])
        if target.exists() and target.read_bytes() != data:
            raise ConfigurationError('Existing integration cache differs: ' + row['target'])
        pending.append((row, target, data))
    for row, target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as output:
                output.write(data)
        if target.read_bytes() != data:
            raise ConfigurationError('Integration write verification failed: ' + row['target'])
    result = dict(status='passed', recipe_sha256=hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
                  files=[dict(target=row['target'], sha256=row['output_sha256']) for row, _, _ in pending],
                  board_access=False)
    atomic_json(resolve_path('vendor://integration_receipt.json'), result)
    return result
