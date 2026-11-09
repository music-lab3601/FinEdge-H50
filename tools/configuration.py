"""Resolve migrated templates and inspect the installed resource configuration."""
import ast
import json
import re

from shared.paths import ROOT, load_settings, resolve_path, work_path

URI = re.compile(r'(?:repo|work|data|vendor|tool)://[^\s\x27\x22{}<>;,\)\]`]+')


def render_text(text):
    return URI.sub(lambda match: resolve_path(match.group()).as_posix(), text)


def render_templates():
    records = json.loads((ROOT / 'provenance/template_migration.json').read_text('utf-8'))
    output = work_path('rendered')
    results = []
    for row in records['files']:
        path = ROOT / row['new']
        text = path.read_text('utf-8-sig')
        rendered = render_text(text)
        if re.search(r'@[A-Z][A-Z0-9_]*@', rendered):
            raise ValueError('Unresolved configuration token in ' + row['new'])
        target = output / row['new']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding='utf-8')
        results.append(dict(path=row['new'], references=len(URI.findall(text)),
                            status='rendered', mode='historical_reference' if
                            row['new'].startswith('experiments/') else 'active_or_build_input'))
    result = dict(status='passed', original_templates=records['original_files'],
                  rendered_templates=len(results), files=results,
                  external_assets_checked_separately=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def check_configuration():
    templates = render_templates()
    external = json.loads((ROOT / 'external/dependencies.json').read_text('utf-8'))['files']
    missing = sorted({row['target'] for row in external if not resolve_path(row['target']).is_file()})
    return dict(status='passed', templates=templates['rendered_templates'],
                configured=load_settings(), external_missing=len(missing),
                external_missing_examples=missing[:12], full_build_ready=not missing,
                board_access=False)


def check_imports():
    errors = []
    count = 0
    for path in ROOT.rglob('*.py'):
        if '.git' in path.parts or '.local' in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text('utf-8-sig'))
        except SyntaxError as exc:
            errors.append(dict(path=path.relative_to(ROOT).as_posix(), error=str(exc)))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in {
                'software', 'host', 'board', 'shared', 'contracts', 'deployment', 'experiments', 'hardware'}:
                module = ROOT.joinpath(*node.module.split('.'))
                if not module.is_dir() and not module.with_suffix('.py').is_file():
                    errors.append(dict(path=path.relative_to(ROOT).as_posix(), missing_module=node.module))
        count += 1
    return dict(status='passed' if not errors else 'failed', python_files=count, errors=errors,
                scope='syntax and internal import targets', board_access=False)
