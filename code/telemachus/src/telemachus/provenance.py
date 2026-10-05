from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, p)


def code_manifest(root):
    root = Path(root)
    files = sorted(root.glob("src/**/*.py"))+sorted(root.glob("tests/*.py"))+sorted(root.glob("scripts/*.py"))+sorted(root.glob("configs/*.json"))+sorted(root.glob("schemas/*.json"))
    for name in ("pyproject.toml","requirements-tested.txt"):
        if (root/name).exists():files.append(root/name)
    return {str(p.relative_to(root)): file_hash(p) for p in files}


def environment_info():
    versions = {}
    for name in ["numpy", "scipy", "torch", "torchvision", "pandas", "pillow", "scikit-learn", "transformers", "sionna", "pytest"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "versions": versions}


def keyed_rng(seed: int, index: int, stream: int, subindex: int = 0):
    """Independent streams: policy RNG cannot mutate physical random streams."""
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(index), int(stream), int(subindex)]))
