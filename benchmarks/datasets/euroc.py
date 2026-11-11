"""EuRoC MAV adapter (phase 15): normalizes EuRoC sequences into the
harness contract. ETH now hosts EuRoC as per-ENVIRONMENT bundles on
the Research Collection (DOI 10.3929/ethz-b-000690084), so this
adapter downloads a whole bundle ONCE and PARTIALLY EXTRACTS just the
wanted sequence (left camera + groundtruth) — the bundle zip is kept
so sibling sequences can be extracted later without re-downloading.

ASL-format facts this adapter encodes (verified against the dataset
layout at write time):
- images: mav0/cam0/data/<ns_timestamp>.png, already RECTIFIED
  (ORB-SLAM's community config uses sensor.yaml's camera_matrix with
  ZERO distortion on the shipped pngs — so calibration.txt carries
  fx fy cx cy and no k1);
- groundtruth: mav0/state_groundtruth_estimate0/data.csv —
  '#'-commented CSV, columns timestamp[ns], p_x p_y p_z, then the
  quaternion in w-x-y-z ORDER (scipy wants x-y-z-w: reordered);
- the state is the IMU/BODY pose in the world frame; sensor.yaml's
  T_BS is the sensor->body transform, so the CAMERA pose is
  T_w_cam = T_w_body @ inv(T_BS).

Layout tolerance: the bundle may contain the sequence's mav0 tree
directly or a nested per-sequence zip — both are handled.
"""
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
EUROC_ROOT = REPO_ROOT / "data" / "euroc"

MAX_TIMESTAMP_DIFF_NS = 20_000_000   # 20 ms, the TUM adapter's tolerance

EUROC_SEQUENCES = {
    "euroc_v1_01_easy": {
        # DSpace 7 serves content at the API endpoint — the public
        # /bitstreams/<uuid>/download link is an Angular route that
        # returns the 782-byte SPA shell to non-browser clients.
        "bundle_url": "https://www.research-collection.ethz.ch/"
                      "server/api/core/bitstreams/"
                      "02ecda9a-298f-498b-970c-b7c44334d880/content",
        "bundle_zip": "euroc_vicon_room1.zip",   # ~5.8 GB (V1_01-03)
        "sequence": "V1_01_easy",
        "cap": 1000,
    },
    "euroc_mh_01_easy": {
        "bundle_url": "https://www.research-collection.ethz.ch/"
                      "server/api/core/bitstreams/"
                      "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content",
        "bundle_zip": "euroc_machine_hall.zip",  # ~12.1 GB (MH_01-05)
        "sequence": "MH_01_easy",
        "cap": 1000,
    },
}

ZIP_MAGIC = b"PK\x03\x04"


def _download(url: str, dst: Path, retries: int = 10):
    """Range-RESUMABLE download to .part then rename, with a magic-
    byte guard (the research collection serves an HTML SPA shell to
    wrong-URL requests — that must fail loudly, not masquerade as a
    zip). 429s back off exponentially (Retry-After honored), so a
    rate-limited fetch waits its turn instead of hammering."""
    if dst.exists() and dst.stat().st_size > 0:
        with open(dst, "rb") as f:
            if f.read(4) != ZIP_MAGIC:
                dst.unlink()
                raise IOError(f"{dst} is not a zip (HTML stub?) — removed")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(retries):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0 (benchmark dataset "
                                     "fetch)"}
            if start:
                headers["Range"] = f"bytes={start}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                resumed = (resp.status == 206)
                if start and not resumed:
                    start = 0     # server ignored the range: restart
                mode = "ab" if (start and resumed) else "wb"
                total = int(resp.headers.get("Content-Length", 0)) + start
                done = start
                with open(part, mode) as f:
                    while True:
                        block = resp.read(4 << 20)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        if total and (done // (100 << 20)) != \
                                ((done - len(block)) // (100 << 20)):
                            print(f"\r[euroc] {dst.name}: "
                                  f"{done / 1e9:.1f}/"
                                  f"{total / 1e9:.1f} GB",
                                  end="", flush=True)
            print("", flush=True)
            if total and done != total:
                raise IOError(f"short download {done}/{total}")
            with open(part, "rb") as f:
                if f.read(4) != ZIP_MAGIC:
                    raise IOError("downloaded content is not a zip "
                                  "(HTML stub?)")
            part.rename(dst)
            return dst
        except Exception as e:          # noqa: BLE001 - retry+report
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:
                # Range past EOF: the .part is ALREADY the complete
                # file (a prior attempt died between the last byte
                # and the rename — measured: 7 x 600 s stalls on the
                # desktop because the retry loop kept re-asking for
                # bytes past the end). Finalize, do not retry.
                with open(part, "rb") as f:
                    if f.read(4) == ZIP_MAGIC:
                        part.rename(dst)
                        print(f"\n[euroc] {dst.name}: resumed .part "
                              "was already complete — finalized",
                              flush=True)
                        return dst
                part.unlink()          # non-zip content: start over
                raise IOError("416 with a non-zip .part — discarded")
            if attempt == retries - 1:
                raise
            wait = min(60 * (2 ** attempt), 600)
            headers = getattr(e, "headers", None)
            if headers is not None:
                try:
                    wait = max(wait, int(headers.get("Retry-After", 0)))
                except (TypeError, ValueError):
                    pass
            kept = (part.stat().st_size / 1e9
                    if part.exists() else 0.0)
            print(f"\n[euroc] download attempt {attempt + 1} failed "
                  f"({e}); retrying in {wait} s "
                  f"({kept:.2f} GB kept for resume)", flush=True)
            time.sleep(wait)


def _open_sequence_zip(bundle_path: Path, sequence: str):
    """(zipfile, prefix, names, temp_path) for the sequence inside the
    bundle — handling both a nested per-sequence zip (extracted to a
    temp FILE, not RAM: the inner zips are ~1.3 GB) and a direct
    mav0 tree. The caller owns deleting temp_path when done."""
    bundle = zipfile.ZipFile(bundle_path)
    nested = [m for m in bundle.namelist()
              if m.endswith(".zip") and sequence in Path(m).name]
    temp_path = None
    if nested:
        temp_path = bundle_path.with_name(f"_{sequence}.inner.zip")
        print(f"[euroc] extracting nested {nested[0]} ...", flush=True)
        with bundle.open(nested[0]) as src, \
                open(temp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, 4 << 20)
        zf = zipfile.ZipFile(temp_path)
        names = zf.namelist()
    else:
        zf = bundle
        names = bundle.namelist()
    # the mav0 root: the common prefix above cam0/data entries
    cam0 = sorted(n for n in names if "/cam0/data/" in n)
    if not cam0:
        raise RuntimeError(f"no cam0 data for {sequence} in bundle")
    prefix = cam0[0].split("cam0/data/")[0]
    return zf, prefix, names, temp_path


def _read_sensor_yaml(zf, names, prefix) -> dict:
    member = prefix + "cam0/sensor.yaml"
    if member not in names:
        raise RuntimeError(f"missing {member}")
    text = zf.read(member).decode()
    # ASL yamls may carry an OpenCV '%YAML:1.0' header pyyaml rejects.
    text = "\n".join(l for l in text.splitlines()
                     if not l.startswith("%YAML"))
    doc = yaml.safe_load(text)
    sensor = doc.get("sensor0", doc)
    fu, fv, cu, cv = (float(v) for v in sensor["intrinsics"][:4])
    t_bs_spec = sensor["T_BS"]
    if isinstance(t_bs_spec, dict):   # {cols: 4, rows: 4, data: [...]}
        t_bs = np.array(t_bs_spec["data"], dtype=float).reshape(
            t_bs_spec["rows"], t_bs_spec["cols"])
    else:                             # flat row-major list
        t_bs = np.array(t_bs_spec, dtype=float).reshape(4, 4)
    return {"fx": fu, "fy": fv, "cx": cu, "cy": cv,
            "T_body_sensor": t_bs}


def _read_groundtruth(zf, names, prefix) -> list:
    """[(ts_ns, T_w_body 4x4)] from the state groundtruth CSV."""
    gt_members = [n for n in names
                  if "state_groundtruth" in n and n.endswith(".csv")]
    if not gt_members:
        raise RuntimeError(f"no state_groundtruth csv under {prefix}")
    rows = []
    for line in zf.read(gt_members[0]).decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("timestamp"):
            continue
        f = line.split(",")
        if len(f) < 8:
            continue
        ts_ns = int(float(f[0]))
        p = np.array([float(f[1]), float(f[2]), float(f[3])])
        # csv order is w x y z; scipy wants x y z w
        q = Rotation.from_quat([float(f[5]), float(f[6]),
                                float(f[7]), float(f[4])])
        T = np.eye(4)
        T[:3, :3] = q.as_matrix()
        T[:3, 3] = p
        rows.append((ts_ns, T))
    rows.sort(key=lambda r: r[0])
    return rows


def prepare(name: str, force: bool = False) -> Path:
    spec = EUROC_SEQUENCES[name]
    norm_dir = EUROC_ROOT / name
    if (norm_dir / "trajectory.txt").exists() and not force:
        return norm_dir

    bundle_path = _download(spec["bundle_url"],
                            EUROC_ROOT / spec["bundle_zip"])
    print(f"[euroc] extracting {spec['sequence']} from "
          f"{bundle_path.name} (partial: cam0 + groundtruth only) ...",
          flush=True)
    zf, prefix, names, inner_temp = _open_sequence_zip(
        bundle_path, spec["sequence"])
    try:
        sensor = _read_sensor_yaml(zf, names, prefix)
        gt = _read_groundtruth(zf, names, prefix)

        T_sensor_body = np.linalg.inv(sensor["T_body_sensor"])
        img_members = sorted(n for n in names
                             if n.startswith(prefix + "cam0/data/")
                             and n.endswith(".png"))
        if not img_members:
            raise RuntimeError("no cam0 images found")

        # the raw extracted tree (kept; frames symlink into it)
        raw_dir = EUROC_ROOT / (spec["sequence"] + "_raw")
        if raw_dir.exists() and force:
            shutil.rmtree(raw_dir)
        (raw_dir / "data").mkdir(parents=True, exist_ok=True)

        frames_dir = norm_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.iterdir():
            old.unlink()

        matched = skipped = 0
        gt_idx = 0
        with open(norm_dir / "trajectory.txt", "w") as traj:
            for member in img_members:
                if matched >= spec["cap"]:
                    break
                ts_img = int(Path(member).stem)
                while (gt_idx + 1 < len(gt)
                        and abs(gt[gt_idx + 1][0] - ts_img)
                        <= abs(gt[gt_idx][0] - ts_img)):
                    gt_idx += 1
                ts_gt, T_w_body = gt[gt_idx]
                if abs(ts_gt - ts_img) > MAX_TIMESTAMP_DIFF_NS:
                    skipped += 1
                    continue

                src = raw_dir / "data" / (Path(member).name)
                if not src.exists():
                    src.write_bytes(zf.read(member))
                dst = frames_dir / f"frame_{matched:06d}.png"
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(src, dst)

                # T_w_cam = T_w_body @ inv(T_BS)
                T_w_cam = T_w_body @ T_sensor_body
                q = Rotation.from_matrix(T_w_cam[:3, :3]).as_quat()  # xyzw
                t = T_w_cam[:3, 3]
                traj.write(f"{matched} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                           f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} "
                           f"{q[3]:.9f}\n")
                matched += 1

        with open(norm_dir / "calibration.txt", "w") as f:
            f.write(f"{sensor['fx']} {sensor['fy']} "
                    f"{sensor['cx']} {sensor['cy']}\n")

        if matched == 0:
            raise RuntimeError(
                f"No EuRoC frames associated with GT for {name}")
        print(f"[euroc] {name}: normalized {matched} frames "
              f"({skipped} skipped: no GT within "
              f"{MAX_TIMESTAMP_DIFF_NS / 1e6:.0f} ms"
              + (f", capped at {spec['cap']}" if spec["cap"] else "")
              + f") -> {norm_dir}", flush=True)
        return norm_dir
    finally:
        zf.close()
        if inner_temp is not None and inner_temp.exists():
            inner_temp.unlink()   # the raw tree carries what we keep
