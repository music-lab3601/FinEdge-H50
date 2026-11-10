"""Verify local documentation, literal resources, HDL and native source references."""
import ast
from pathlib import Path
import re
from urllib.parse import unquote


def check(root, tracked):
    root = root.resolve()
    tracked = set(tracked)
    errors = []
    checked = 0

    def require(path, source):
        nonlocal checked
        checked += 1
        path = path.resolve()
        if not path.is_relative_to(root):
            errors.append('Local reference escapes repository in ' + source)
        elif not path.exists():
            errors.append('Missing local reference in ' + source + ': ' + path.relative_to(root).as_posix())
        elif path.is_file() and path.relative_to(root).as_posix() not in tracked:
            errors.append('Referenced file absent from Git in ' + source + ': ' + path.relative_to(root).as_posix())

    headers = {Path(name).name: root/name for name in tracked if Path(name).suffix in {'.h', '.hpp'}}
    active = {'hardware', 'board', 'host', 'deployment', 'software', 'tools', 'shared'}
    for name in sorted(tracked):
        path = root/name
        if not path.is_file():
            continue
        if path.suffix not in {'.md', '.py', '.sv', '.v', '.cpp', '.c', '.h', '.hpp'} and path.name != 'CMakeLists.txt':
            continue
        text = path.read_text('utf-8-sig')
        if path.suffix == '.md':
            for link in re.findall(r'\]\(([^)]+)\)', text):
                link = unquote(link.split(' ', 1)[0].strip('<>').split('#', 1)[0])
                if link and not re.match(r'[a-zA-Z]+:', link):
                    require(path.parent/link, name)
        if path.suffix == '.py' and Path(name).parts[0] in active:
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('repo://'):
                    require(root/node.value.removeprefix('repo://'), name)
        if name.startswith('hardware/rtl/') and path.suffix in {'.sv', '.v'}:
            for include, rom in re.findall(r'`include\s+"([^"]+)"|"([^"\n]+\.hex)"', text):
                value = include or rom
                target = root/value.removeprefix('repo://') if value.startswith('repo://') else path.parent/value
                if not target.is_file():
                    target = root/'hardware/rom'/value
                require(target, name)
        if path.name == 'CMakeLists.txt':
            for value in re.findall(r'(?<![\w/])([\w./-]+\.(?:cpp|c))\b', text):
                require(path.parent/value, name)
        if path.suffix in {'.cpp', '.c', '.h', '.hpp'} and Path(name).parts[0] in active:
            for value in re.findall(r'^\s*#include\s+"([^"]+)"', text, re.MULTILINE):
                target = path.parent/value
                if not target.is_file() and value in headers:
                    target = headers[value]
                require(target, name)
    return dict(status='passed' if not errors else 'failed', references_checked=checked, errors=errors)
