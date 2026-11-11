"""The GT-ladder registry (phase 15): per-sequence metadata + the
prepare() dispatch. One sequence per insight axis (the selection
rationale is recorded in IMPROVEMENT.md's phase-15 plan).

Sequence tiers:
- 'tum_fr1_xyz' REUSES the existing battery preparation
  (benchmarks/tum.py) — the same underlying data, one download, and
  direct comparability to today's numbers.
- The other TUM entries are prepared by the generalized adapter
  (tum_rgbd.py).
- EuRoC entries are prepared by the bundle adapter (euroc.py): ETH
  now hosts EuRoC as per-ENVIRONMENT bundles (Vicon Room 1 = 5.8 GB
  for three sequences; Machine Hall = 12.1 GB for five), so the
  adapter PARTIALLY EXTRACTS just the wanted sequence's cam0 +
  groundtruth from the bundle zip.

KITTI: DROPPED by user decision (2026-11-11) — the official odometry
downloads sit behind cvlibs registration on every path, bypassing a
registration wall would circumvent the dataset's access terms, and
no legitimate per-sequence mirror exists (the Ultralytics mirror is
the object-detection subset, no pose GT). The driving/scale-drift
axis is dropped with it; a kitti.py adapter remains a ~1-hour add
if the user ever registers.

Usage:
    uv run python -m benchmarks.datasets info
    uv run python -m benchmarks.datasets prepare --all
    uv run python -m benchmarks.datasets prepare --sequences tum_fr1_desk
"""
import argparse
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

# The axis rationale strings mirror IMPROVEMENT.md's phase-15 plan.
GT_SEQUENCES = {
    "tum_fr1_xyz": {
        "adapter": "benchmarks.tum (the existing battery entry)",
        "source": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
                  "rgbd_dataset_freiburg1_xyz.tgz",
        "license": "CC-BY 3.0 (TUM RGB-D); local research use",
        "axis": "anchor: comparability to the existing battery numbers",
        "cap": None,
    },
    "tum_fr2_xyz": {
        "adapter": "datasets.tum_rgbd",
        "source": "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/"
                  "rgbd_dataset_freiburg2_xyz.tgz",
        "license": "CC-BY 3.0 (TUM RGB-D); local research use",
        "axis": "sustained steady translation: long-horizon drift",
        "cap": 800,
    },
    "tum_fr1_desk": {
        "adapter": "datasets.tum_rgbd",
        "source": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
                  "rgbd_dataset_freiburg1_desk.tgz",
        "license": "CC-BY 3.0 (TUM RGB-D); local research use",
        "axis": "fast handheld + loop on fr1's rolling-shutter "
                "photometry",
        "cap": None,
    },
    "tum_fr3_long_office_household": {
        "adapter": "datasets.tum_rgbd",
        "source": "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/"
                  "rgbd_dataset_freiburg3_long_office_household.tgz",
        "license": "CC-BY 3.0 (TUM RGB-D); local research use",
        "axis": "the classic long real loop: what a loop-closure "
                "correction BUYS against GT (KF-ATE before/after)",
        "cap": None,   # FULL by user decision — the loop is the point
    },
    "euroc_v1_01_easy": {
        "adapter": "datasets.euroc",
        "source": "ETH Research Collection, DOI 10.3929/ethz-b-000690084 "
                  "(Vicon Room 1 bundle, ~5.8 GB; the sequence's share "
                  "~1.3 GB is partially extracted)",
        "license": "CC-BY-SA (EuRoC MAV, ETH Zurich); local research use",
        "axis": "MAV rotation+translation: the phase-13 regime with GT",
        "cap": 1000,
    },
    "euroc_mh_01_easy": {
        "adapter": "datasets.euroc",
        "source": "ETH Research Collection, DOI 10.3929/ethz-b-000690084 "
                  "(Machine Hall bundle, ~12.1 GB; the sequence's share "
                  "~1.3 GB is partially extracted)",
        "license": "CC-BY-SA (EuRoC MAV, ETH Zurich); local research use",
        "axis": "the machine hall's texture-poor content: the "
                "finding-35/36 recovery machinery under GT",
        "cap": 1000,
    },
}


def prepare(name: str, force: bool = False):
    """Download + normalize one ladder sequence -> the dataset dir the
    harness consumes. Idempotent (returns early when prepared)."""
    if name not in GT_SEQUENCES:
        raise ValueError(f"unknown GT-ladder sequence '{name}'. "
                         f"Available: {sorted(GT_SEQUENCES)}")
    if name == "tum_fr1_xyz":
        from benchmarks.tum import prepare_tum_sequence
        return prepare_tum_sequence(force=force)

    from benchmarks.datasets import euroc, tum_rgbd
    if name in tum_rgbd.TUM_SEQUENCES:
        return tum_rgbd.prepare(name, force=force)
    if name in euroc.EUROC_SEQUENCES:
        return euroc.prepare(name, force=force)
    raise RuntimeError(f"no adapter wired for '{name}'")


def describe():
    lines = []
    for name, m in GT_SEQUENCES.items():
        cap = f"cap {m['cap']}" if m["cap"] else "full"
        lines.append(f"{name:34s} [{cap}]  {m['axis']}")
        lines.append(f"{'':34s} source: {m['source']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["info", "prepare"])
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "info":
        print(describe())
        return 0

    names = list(GT_SEQUENCES) if args.all else (args.sequences or [])
    if not names:
        parser.error("prepare needs --all or --sequences NAME ...")
    for name in names:
        print(f"[datasets] preparing {name} ...", flush=True)
        seq_dir = prepare(name, force=args.force)
        print(f"[datasets] {name} ready: {seq_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
