"""Admit original deliveries without executing provenance or historical scripts."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import zipfile

from .paths import ARTIFACTS, MANIFEST_SHA, MODEL_ZIP, PACKAGE, RAW, RAW_MANIFEST_SHA, RAW_ZIP


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe delivery path: {name}")
    root = Path(root).resolve()
    target = root.joinpath(*rel.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError("Delivery path escaped root")
    return target


def put_original(root, name, content):
    target = safe_path(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite modified original: {target}")
    else:
        with target.open("xb") as f:
            f.write(content)


def admit_deliveries():
    with zipfile.ZipFile(MODEL_ZIP) as z:
        manifest_bytes = z.read("package/package_manifest.json")
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA:
            raise ValueError("Model manifest differs from the delivered ED4 identity")
        manifest = json.loads(manifest_bytes)
        for name, expected in manifest["files"].items():
            b = z.read("package/" + name)
            if len(b) != expected["bytes"] or hashlib.sha256(b).hexdigest() != expected["sha256"]:
                raise ValueError(f"Model ZIP integrity failure: {name}")
        for name in z.namelist():
            if name.startswith("package/") and not name.endswith("/"):
                put_original(PACKAGE, name[len("package/"):], z.read(name))
    with zipfile.ZipFile(RAW_ZIP) as z:
        prefix = z.namelist()[0].split("/")[0] + "/"
        manifest_bytes = z.read(prefix + "manifest.json")
        if hashlib.sha256(manifest_bytes).hexdigest() != RAW_MANIFEST_SHA:
            raise ValueError("Raw manifest differs from the RAW1024 delivery")
        raw_manifest = json.loads(manifest_bytes)
        if raw_manifest["model_manifest_sha256"] != MANIFEST_SHA:
            raise ValueError("Raw delivery belongs to a different model")
        for line in z.read(prefix + "SHA256SUMS.txt").decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(None, 1)
            name = name.strip()
            b = z.read(prefix + name)
            if hashlib.sha256(b).hexdigest() != digest:
                raise ValueError(f"Raw ZIP hash failure: {name}")
        # Keep data and reference documents; do not import/run provenance programs.
        selected = {"manifest.json", "SHA256SUMS.txt", "README_先读我.txt", "row_metadata.jsonl",
                    "raw_book.u64le.bin", "normalized_rows.f32le.bin", "source_slice.records.bin",
                    "screening_reject_masks.u32le.bin", "window_inputs.f32le.bin",
                    "window_logits.f32le.bin", "window_classes.u8.bin", "windows.json",
                    "reports/acceptance.json", "provenance/source_schema.json"}
        for name in selected:
            put_original(RAW, name, z.read(prefix + name))
    result = {"status": "passed", "model_manifest_sha256": MANIFEST_SHA,
              "raw_manifest_sha256": RAW_MANIFEST_SHA,
              "model_zip_sha256": sha256(MODEL_ZIP), "raw_zip_sha256": sha256(RAW_ZIP),
              "model_payload_files": len(manifest["files"]), "raw_records": raw_manifest["record_count"],
              "archive_accessed": False}
    write_json(ARTIFACTS / "reports/delivery_verification.json", result)
    return result


def runtime_module():
    if sha256(PACKAGE / "package_manifest.json") != MANIFEST_SHA:
        raise ValueError("Model package manifest mismatch")
    manifest = read_json(PACKAGE / "package_manifest.json")
    if sha256(PACKAGE / "runtime.py") != manifest["files"]["runtime.py"]["sha256"]:
        raise ValueError("Delivered runtime was modified")
    spec = importlib.util.spec_from_file_location("ed4_delivered_runtime", PACKAGE / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_package(PACKAGE, MANIFEST_SHA)
    return module

