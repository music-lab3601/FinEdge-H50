"""Repository-relative configuration and explicit external resource resolution."""
from pathlib import Path, PurePosixPath
import copy
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/default.json'


class ConfigurationError(ValueError):
    pass


def _merge_settings(overrides):
    result = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    for section, values in overrides.items():
        if section not in result or not isinstance(values, dict):
            raise ConfigurationError('Unknown configuration section: ' + section)
        unknown = set(values) - set(result[section])
        if unknown:
            raise ConfigurationError('Unknown configuration keys: ' + ', '.join(sorted(unknown)))
        result[section].update(values)
    if result['board']['horizon'] != 50:
        raise ConfigurationError('This repository freezes the H50 model horizon at 50')
    return result


def load_settings():
    local = Path(os.environ.get('FINEDGE_CONFIG', ROOT / '.local/config.json'))
    return _merge_settings(json.loads(local.read_text(encoding='utf-8')) if local.is_file() else {})


def _configured_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def work_path(relative=''):
    return _join(_configured_path(load_settings()['paths']['work_root']), relative)


def data_path(relative=''):
    return _join(_configured_path(load_settings()['paths']['data_root']), relative)


def _join(base, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or '..' in part.parts or '\\' in relative or ':' in relative:
        raise ConfigurationError('Resource references must remain inside their declared root')
    path = base.joinpath(*part.parts).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ConfigurationError('Resource reference escapes its configured root')
    return path


def resolve_path(reference):
    if isinstance(reference, Path):
        return reference.resolve()
    if '://' not in reference:
        return _configured_path(reference)
    scheme, relative = reference.split('://', 1)
    settings = load_settings()
    if scheme == 'repo':
        base = ROOT
    elif scheme == 'work':
        base = _configured_path(settings['paths']['work_root'])
    elif scheme == 'data':
        base = _configured_path(settings['paths']['data_root'])
    elif scheme == 'vendor':
        base = _configured_path(settings['paths']['vendor_root'])
    elif scheme == 'tool':
        tool, _, relative = relative.partition('/')
        if tool not in settings['tools']:
            raise ConfigurationError('Unknown tool: ' + tool)
        value = settings['tools'][tool]
        base = _configured_path(value) if value else work_path('toolchains/' + tool)
        if base.suffix.lower() in {'.exe', '.bat', '.cmd'}:
            base = base.parent.parent
    else:
        raise ConfigurationError('Unsupported resource scheme: ' + scheme)
    return _join(base, relative)


def resolve_references(value):
    if isinstance(value, str) and value.startswith(('repo://', 'work://', 'data://', 'vendor://', 'tool://')):
        return resolve_path(value).as_posix()
    if isinstance(value, list):
        return [resolve_references(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve_references(item) for key, item in value.items()}
    return value


def read_config(path):
    return resolve_references(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def tool_executable(name, required=True):
    settings = load_settings()
    value = settings['tools'].get(name)
    if value:
        path = _configured_path(value)
        if path.is_file():
            return path
        suffix = '.bat' if name == 'vivado' and os.name == 'nt' else '.exe' if os.name == 'nt' else ''
        for candidate in (path / 'bin' / (name + suffix), path / (name + suffix)):
            if candidate.is_file():
                return candidate
    found = shutil.which(name)
    if found:
        return Path(found).resolve()
    if required:
        raise ConfigurationError('Configure tool ' + name + ' with finedge.py configure --' + name)
    return None


def environment():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    icraft = resolve_path('tool://icraft/bin')
    if icraft.is_dir():
        env['PATH'] = str(icraft) + os.pathsep + env.get('PATH', '')
    return env


def source_path(old_relative):
    """Resolve a recorded source identity while historical helper code is retained."""
    aliases = json.loads((ROOT / 'provenance/layout_map.json').read_text(encoding='utf-8'))['files']
    target = aliases.get(old_relative, old_relative)
    path = _join(ROOT, target)
    if not path.is_file():
        raise ConfigurationError('Source identity has no collected file: ' + old_relative)
    return path


def save_configuration(changes):
    path = ROOT / '.local/config.json'
    current = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for section, values in changes.items():
        current.setdefault(section, {}).update(values)
    _merge_settings(current)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return path
