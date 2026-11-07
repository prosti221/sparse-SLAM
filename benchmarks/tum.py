"""TUM RGB-D dataset adapter for the benchmark harness.

Downloads the freiburg1/xyz sequence on demand, associates RGB frames
with ground-truth poses by timestamp, and normalizes everything into the
same directory layout the synthetic generator produces:

    data/tum/tum_xyz/
        frames/frame_%06d.png   symlinks into the raw dataset
        trajectory.txt          "i tx ty tz qx qy qz qw" (Twc, TUM format)
        calibration.txt         "fx fy cx cy" (freiburg1 factory values)

No scene_points/scene_planes: real imagery has no analytic scene geometry,
so map-vs-scene metrics are skipped for this sequence.

The system itself is monocular - the RGB-D depth data is ignored (a future
milestone could use it for map evaluation).
"""
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TUM_ROOT = REPO_ROOT / "data" / "tum"
RAW_DIR = TUM_ROOT / "rgbd_dataset_freiburg1_xyz"
NORM_DIR = TUM_ROOT / "tum_xyz"

TUM_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"

# Freiburg 1 RGB camera factory calibration (per TUM dataset website)
FX, FY, CX, CY = 517.306409, 516.469215, 318.643040, 255.313989

MAX_TIMESTAMP_DIFF = 0.02  # seconds; same as TUM's associate.py default


def _read_stamped_list(path: Path):
    """Parse a TUM stamped file: '# comment' lines skipped, rest is
    'timestamp field1 field2 ...'. Returns [(timestamp, [fields])]."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        entries.append((float(parts[0]), parts[1:]))
    return entries


def _download_progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0 and block_num % 200 == 0:
        print(f"\r[tum] downloaded {done / 1e6:.0f}/{total_size / 1e6:.0f} MB",
              end="", flush=True)


def ensure_downloaded() -> Path:
    """Download and extract the raw dataset if not present."""
    if (RAW_DIR / "groundtruth.txt").exists():
        return RAW_DIR

    TUM_ROOT.mkdir(parents=True, exist_ok=True)
    tgz = TUM_ROOT / "rgbd_dataset_freiburg1_xyz.tgz"
    print(f"[tum] downloading {TUM_URL} (~450 MB, first run only)...")
    urllib.request.urlretrieve(TUM_URL, tgz, reporthook=_download_progress)
    print("\n[tum] extracting...")
    with tarfile.open(tgz) as tar:
        tar.extractall(TUM_ROOT)
    tgz.unlink()
    print(f"[tum] raw dataset ready: {RAW_DIR}")
    return RAW_DIR


def prepare_tum_sequence(force: bool = False) -> Path:
    """Normalize the raw TUM sequence into the benchmark dataset layout."""
    ensure_downloaded()

    if (NORM_DIR / "trajectory.txt").exists() and not force:
        return NORM_DIR

    rgb_entries = _read_stamped_list(RAW_DIR / "rgb.txt")
    gt_entries = _read_stamped_list(RAW_DIR / "groundtruth.txt")
    # Both lists are sorted by timestamp (TUM guarantees this).
    rgb_entries.sort(key=lambda e: e[0])
    gt_entries.sort(key=lambda e: e[0])

    frames_dir = NORM_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.iterdir():
        old.unlink()

    matched = skipped = 0
    gt_idx = 0
    with open(NORM_DIR / "trajectory.txt", "w") as traj:
        for ts_rgb, fields in rgb_entries:
            rel_path = fields[0]
            # Advance the ground-truth pointer to the closest timestamp.
            while (gt_idx + 1 < len(gt_entries)
                   and abs(gt_entries[gt_idx + 1][0] - ts_rgb)
                   <= abs(gt_entries[gt_idx][0] - ts_rgb)):
                gt_idx += 1
            ts_gt, gt_fields = gt_entries[gt_idx]
            if abs(ts_gt - ts_rgb) > MAX_TIMESTAMP_DIFF or len(gt_fields) < 7:
                skipped += 1
                continue

            src = (RAW_DIR / rel_path).resolve()
            if not src.exists():
                skipped += 1
                continue
            dst = frames_dir / f"frame_{matched:06d}.png"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)

            tx, ty, tz, qx, qy, qz, qw = (float(v) for v in gt_fields[:7])
            traj.write(f"{matched} {tx:.9f} {ty:.9f} {tz:.9f} "
                       f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            matched += 1

    with open(NORM_DIR / "calibration.txt", "w") as f:
        f.write(f"{FX} {FY} {CX} {CY}\n")

    if matched == 0:
        raise RuntimeError("No TUM frames could be associated with ground truth")

    print(f"[tum] normalized {matched} frames "
          f"({skipped} skipped: no GT within {MAX_TIMESTAMP_DIFF}s) -> {NORM_DIR}")
    return NORM_DIR


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_tum_sequence(force=args.force)
