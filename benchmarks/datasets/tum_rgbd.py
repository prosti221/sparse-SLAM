"""Generalized TUM RGB-D adapter (phase 15): the freiburg sequences
beyond the battery's existing tum_xyz entry, normalized into the same
on-disk contract (frames/frame_%06d.png symlinks + TUM-format
trajectory.txt + calibration.txt with the per-freiburg factory
values). The association logic is the battery adapter's own
(pointer-advance nearest-GT match, 20 ms tolerance) so every ladder
entry associates exactly like the anchor.

fr3_long_office_household runs FULL by user decision (the loop is
the point); fr2_xyz caps at 800 frames (the sustained-drift axis does
not need the tail).
"""
import os
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TUM_ROOT = REPO_ROOT / "data" / "tum"

MAX_TIMESTAMP_DIFF = 0.02  # seconds; the battery adapter's tolerance

# Factory calibrations (TUM dataset website).
CALIB = {
    "freiburg1": (517.306409, 516.469215, 318.643040, 255.313989),
    "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}

TUM_SEQUENCES = {
    "tum_fr2_xyz": {
        "url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/"
               "rgbd_dataset_freiburg2_xyz.tgz",
        "raw_dir": "rgbd_dataset_freiburg2_xyz",
        "calib": "freiburg2",
        "cap": 800,
    },
    "tum_fr1_desk": {
        "url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
               "rgbd_dataset_freiburg1_desk.tgz",
        "raw_dir": "rgbd_dataset_freiburg1_desk",
        "calib": "freiburg1",
        "cap": None,
    },
    "tum_fr3_long_office_household": {
        "url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/"
               "rgbd_dataset_freiburg3_long_office_household.tgz",
        "raw_dir": "rgbd_dataset_freiburg3_long_office_household",
        "calib": "freiburg3",
        "cap": None,   # FULL — the loop is the point (user decision)
    },
}


def _read_stamped_list(path: Path):
    """Parse a TUM stamped file ('# comment' lines skipped)."""
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        entries.append((float(parts[0]), parts[1:]))
    return entries


def _download_progress(block_num, block_size, total_size):
    if total_size > 0 and block_num % 400 == 0:
        print(f"\r[tum] downloaded "
              f"{block_num * block_size / 1e6:.0f}/"
              f"{total_size / 1e6:.0f} MB", end="", flush=True)


def _ensure_downloaded(spec) -> Path:
    raw_dir = TUM_ROOT / spec["raw_dir"]
    if (raw_dir / "groundtruth.txt").exists():
        return raw_dir
    TUM_ROOT.mkdir(parents=True, exist_ok=True)
    tgz = TUM_ROOT / (spec["raw_dir"] + ".tgz")
    if not tgz.exists():
        print(f"[tum] downloading {spec['url']} (first run only)...",
              flush=True)
        urllib.request.urlretrieve(spec["url"], tgz,
                                   reporthook=_download_progress)
        print("", flush=True)
    print(f"[tum] extracting {tgz.name} ...", flush=True)
    with tarfile.open(tgz) as tar:
        tar.extractall(TUM_ROOT)
    tgz.unlink()   # raw frames stay (the normalized symlinks point in)
    return raw_dir


def prepare(name: str, force: bool = False) -> Path:
    spec = TUM_SEQUENCES[name]
    norm_dir = TUM_ROOT / name
    if (norm_dir / "trajectory.txt").exists() and not force:
        return norm_dir

    raw_dir = _ensure_downloaded(spec)

    rgb_entries = _read_stamped_list(raw_dir / "rgb.txt")
    gt_entries = _read_stamped_list(raw_dir / "groundtruth.txt")
    rgb_entries.sort(key=lambda e: e[0])
    gt_entries.sort(key=lambda e: e[0])

    frames_dir = norm_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.iterdir():
        old.unlink()

    matched = skipped = 0
    gt_idx = 0
    with open(norm_dir / "trajectory.txt", "w") as traj:
        for ts_rgb, fields in rgb_entries:
            if spec["cap"] is not None and matched >= spec["cap"]:
                break
            rel_path = fields[0]
            while (gt_idx + 1 < len(gt_entries)
                    and abs(gt_entries[gt_idx + 1][0] - ts_rgb)
                    <= abs(gt_entries[gt_idx][0] - ts_rgb)):
                gt_idx += 1
            ts_gt, gt_fields = gt_entries[gt_idx]
            if abs(ts_gt - ts_rgb) > MAX_TIMESTAMP_DIFF or len(gt_fields) < 7:
                skipped += 1
                continue

            src = (raw_dir / rel_path).resolve()
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

    fx, fy, cx, cy = CALIB[spec["calib"]]
    with open(norm_dir / "calibration.txt", "w") as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")

    if matched == 0:
        raise RuntimeError(f"No frames associated with GT for {name}")
    print(f"[tum] {name}: normalized {matched} frames "
          f"({skipped} skipped, no GT within {MAX_TIMESTAMP_DIFF}s"
          + (f", capped at {spec['cap']}" if spec["cap"] else "")
          + f") -> {norm_dir}", flush=True)
    return norm_dir
