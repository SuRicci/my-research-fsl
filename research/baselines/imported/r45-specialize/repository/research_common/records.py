from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path, data, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Immutable artifact exists: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def snapshot_run(project, output, command, config, inputs=None):
    import numpy as np
    project, output = Path(project), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new run ID: {output}")
    output.mkdir(parents=True)
    files = list((WORKSPACE / "research_common").glob("*.py"))
    files += list(project.glob("*.py")) + list(project.glob("*.json"))
    files += list((project / "src").glob("*.py")) + list((project / "configs").glob("*.json"))
    sources = {}
    for src in sorted(files):
        relative = src.relative_to(WORKSPACE)
        dest = output / "code_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        sources[relative.as_posix()] = sha256(src)
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip()
    except Exception:
        git_head = None
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": command,
              "config": config, "inputs": inputs or {}, "sources": sources,
              "git_head_base_only": git_head, "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "numpy": np.__version__}
    write_json(output / "receipt.json", record)
    return output


def download_public(url, destination, expected_sha256=None):
    import requests
    dest = Path(destination)
    if dest.exists():
        digest = sha256(dest)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(f"Existing checksum mismatch: {dest}")
        return {"url": url, "path": str(dest), "bytes": dest.stat().st_size, "sha256": digest, "existing": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".download")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 45)) as response:
                response.raise_for_status()
                size = int(response.headers.get("Content-Length", 0))
                n, last = 0, time.monotonic()
                with tmp.open("wb") as f:
                    for block in response.iter_content(1024**2):
                        f.write(block)
                        n += len(block)
                        if time.monotonic() - last > 15:
                            print(f"download {dest.name}: {n / 1024**2:.1f} MiB", flush=True)
                            last = time.monotonic()
                if size and not response.headers.get("Content-Encoding") and n != size:
                    raise IOError("Incomplete public download")
            digest = sha256(tmp)
            if expected_sha256 and digest != expected_sha256:
                raise ValueError("Downloaded checksum mismatch")
            os.replace(tmp, dest)
            return {"url": url, "path": str(dest), "bytes": n, "sha256": digest}
        except Exception:
            if attempt == 2:
                raise

