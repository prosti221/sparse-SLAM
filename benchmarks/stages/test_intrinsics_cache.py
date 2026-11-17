"""The intrinsics-cache pins: M1b's provenance flow.

A restart must not re-estimate: previously used sources remember
their intrinsics keyed by content identity (path/size/mtime for a
file, path/count/mtime for a sequence dir); the live camera is never
cached. A manual entry overwrites any estimated one for the identity
(the served row carries the manual provenance).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from slam.core import intrinsics_cache as cache


def _video_file(tmp_path: Path, tag="v"):
    f = tmp_path / f"{tag}.mp4"
    f.write_bytes(b"x" * 128)
    return f


def test_file_identity_changes_with_content(tmp_path):
    f = _video_file(tmp_path)
    a = cache._source_identity(str(f))
    import os
    os.utime(f, (0, 0))  # a different mtime
    b = cache._source_identity(str(f))
    assert a["path"] == b["path"] and a["size"] == b["size"]
    assert a != b  # the mtime change invalidates the cache row


def test_sequence_identity_counts_frames(tmp_path):
    d = tmp_path / "seq"
    d.mkdir()
    for i in range(4):
        (d / f"{i:06d}.png").write_bytes(b"p")
    ident = cache._source_identity(str(d))
    assert ident["kind"] == "sequence" and ident["count"] == 4
    (d / "000004.png").write_bytes(b"p")
    assert cache._source_identity(str(d))["count"] == 5


def test_camera_never_caches():
    assert cache._source_identity("1") is None
    assert cache._source_identity("/nonexistent/path.mp4") is None


def test_store_lookup_forget_round_trip(tmp_path):
    store_path = tmp_path / "cache.json"
    f = _video_file(tmp_path)
    row = cache.store(str(f), "geocalib", cache_path=store_path,
                      fx=900.0, fy=900.0, cx=480.0, cy=270.0, k1=-0.01)
    assert cache_path_has(store_path, f)
    hit = cache.lookup(str(f), cache_path=store_path)
    assert hit["provenance"] == "geocalib"
    K = cache.cached_K(hit)
    assert K.tolist() == [[900.0, 0.0, 480.0],
                          [0.0, 900.0, 270.0],
                          [0.0, 0.0, 1.0]]

    # The manual provenance overwrites the estimated row for the same
    # identity: manual > cache in the precedence order.
    row2 = cache.store(str(f), "manual", cache_path=store_path,
                       fx=1234.0, fy=1234.0,
                       cx=481.0, cy=271.0, k1=0.0)
    assert row2["provenance"] == "manual"
    hit2 = cache.lookup(str(f), cache_path=store_path)
    assert hit2["provenance"] == "manual" and hit2["fx"] == 1234.0

    assert cache.forget(str(f), cache_path=store_path) is True
    assert cache.lookup(str(f), cache_path=store_path) is None
    assert cache.forget(str(f), cache_path=store_path) is False


def cache_path_has(store_path: Path, f: Path):
    rows = json.loads(store_path.read_text())
    return any(r["location"] == str(f) for r in rows.values())


def test_bad_provenance_rejected(tmp_path):
    f = _video_file(tmp_path)
    with pytest.raises(ValueError):
        cache.store(str(f), "wild", fx=1.0, fy=1.0, cx=1.0, cy=1.0,
                    k1=0.0)
