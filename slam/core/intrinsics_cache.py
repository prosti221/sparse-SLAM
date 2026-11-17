"""The intrinsics cache: previously used sources remember their
estimates so a restart skips re-estimation.

Identity key (the load identity must survive restarts AND detect a
changed source, since stale intrinsics on new content are worse than
re-estimating): a file = absolute path + size + mtime; a sequence
dir = absolute path + frame count + dir mtime; the live camera is
never cached. Stored per key: fx/fy/cx/cy/k1, the provenance
(geocalib / config / manual), estimated_at. Precedence manual >
cache > config is carried in the stored provenance: a manual entry
overwrites any estimated one for the identity and is served first.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_PATH = Path("data/ui/intrinsics_cache.json")


def _source_identity(location: str) -> Optional[dict]:
    """None = never cached (the live camera, or unresolvable content)."""
    path = Path(location)
    if path.is_dir():
        frames = list(path.glob("*.png"))
        if not frames:
            return None
        stat = path.stat()
        return {"kind": "sequence", "path": str(path.resolve()),
                "count": len(frames),
                "mtime": int(stat.st_mtime)}
    if path.is_file():
        stat = path.stat()
        return {"kind": "file", "path": str(path.resolve()),
                "size": stat.st_size, "mtime": int(stat.st_mtime)}
    # Either a live camera (an int-able index) or a missing source:
    # neither is cached.
    return None


def _storage_key(identity: dict) -> str:
    blob = json.dumps(identity, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def lookup(location: str, cache_path: Path = None) -> Optional[dict]:
    identity = _source_identity(location)
    if identity is None or not identity.get("path"):
        return None
    store_path = cache_path or CACHE_PATH
    if not store_path.exists():
        return None
    rows = json.loads(store_path.read_text())
    row = rows.get(_storage_key(identity))
    if not row:
        return None
    row = dict(row, identity=identity, key=_storage_key(identity))
    return row


def store(location: str, provenance: str, cache_path: Path = None,
          **intrinsics) -> Optional[dict]:
    """Write-through: `**intrinsics` carries fx/fy/cx/cy/k1 (fx/fy/cx/cy
    may be None for the auto-sentinel values the caller resolved
    later); provenance = 'geocalib' | 'config' | 'manual'."""
    if provenance not in ("geocalib", "config", "manual"):
        raise ValueError(provenance)
    identity = _source_identity(location)
    if identity is None or not identity.get("path"):
        return None
    store_path = cache_path or CACHE_PATH
    store_path.parent.mkdir(parents=True, exist_ok=True)
    rows = (json.loads(store_path.read_text())
            if store_path.exists() else {})
    import time
    key = _storage_key(identity)
    rows[key] = {"provenance": provenance,
                 "estimated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "location": str(location), **intrinsics}
    store_path.write_text(json.dumps(rows, indent=1) + "\n")
    return rows[key]


def forget(location: str, cache_path: Path = None) -> bool:
    identity = _source_identity(location)
    if identity is None or not identity.get("path"):
        return False
    store_path = cache_path or CACHE_PATH
    if not store_path.exists():
        return False
    rows = json.loads(store_path.read_text())
    key = _storage_key(identity)
    if key not in rows:
        return False
    del rows[key]
    store_path.write_text(json.dumps(rows, indent=1) + "\n")
    return True


def cached_K(row: dict) -> np.ndarray:
    fx, fy = row["fx"], row.get("fy", row["fx"])
    cx, cy = row["cx"], row["cy"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                    dtype=float)
