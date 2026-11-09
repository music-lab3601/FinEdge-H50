from pathlib import Path
import hashlib
import json
import os
import tempfile
from shared.paths import ROOT as REPOSITORY_ROOT, resolve_references, work_path

ROOT = work_path('board')
PROJECT = REPOSITORY_ROOT
STAGES = ['small_benchmark', 'feature_gate', 'int8', 'scheduler_pipeline', 'tensor_transport']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return resolve_references(json.loads(Path(path).read_text(encoding='utf-8')))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def contained_file(root, relative):
    """Reject absolute paths, traversal, links and escaping reparse points."""
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or rel.drive or '..' in rel.parts or not rel.parts:
        raise ValueError('File must be relative and remain in its task directory')
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('Links/junctions are not allowed in a submitted file path')
    path = current.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Submitted file missing or outside task directory')
    return path
