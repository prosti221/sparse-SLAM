"""Remote benchmark offloading (2026-11-11): run the long diagnostic
tiers on a LAN desktop instead of the laptop.

  uv run python -m benchmarks.remote probe
  uv run python -m benchmarks.remote sync [--videos F.mp4 ... | --all-videos]
                                          [--data tum | --data euroc] [--weights]
  uv run python -m benchmarks.remote run -- python -m benchmarks.video_census ...
  uv run python -m benchmarks.remote fetch [--force]

Design rules (the discipline this encodes):

- CROSS-MACHINE BIT-IDENTITY IS NOT A THING. The laptop is arm64,
  the desktop x86-64; BLAS/OpenCV float paths differ. The frozen
  battery ADOPTION GATES (bit-identity A/B vs this machine's
  reference reports) stay on the laptop. A/B experiments are valid
  ANYWHERE as long as BOTH LEGS run on the same machine — that is
  the canonical offload shape (threshold sweeps, census legs,
  attribution runs).
- Every fetched artifact carries a provenance manifest (host,
  platform, command, timestamps) into output/benchmarks/remote_runs/
  — remote evidence is never mistaken for local evidence.
- Fetch NEVER overwrites local files (--ignore-existing; the
  evidence discipline). Collisions are reported; --force is
  explicit.
- Content sync is OPT-IN (code always; videos/data/weights only when
  asked). Videos are YouTube-license material: they stay on the
  user's own machines (LAN sync), never further. Datasets are
  CC-BY/-SA research use, same.
- Long runs hold their ssh session (the local background job's
  completion = the run's completion). Detached mode can come later.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "cankourn@arch-desktop.local"
REMOTE_DIR = "~/sparse-SLAM"
MANIFEST_DIR = REPO_ROOT / "output" / "benchmarks" / "remote_runs"

# The NORMALIZED sequence trees only — all a remote run needs: every
# dataset adapter short-circuits on prepared data before any
# download. The normalized frames are SYMLINKS into raw/bundle trees
# locally, so data syncs pass rsync -L (copy the TARGETS as real
# files; openrsync supports it) — the 18 GB EuRoC bundles and the
# raw TUM datasets never cross the wire.
DATA_TREES = {
    "tum": ["data/tum/tum_xyz", "data/tum/tum_fr1_desk",
            "data/tum/tum_fr2_xyz",
            "data/tum/tum_fr3_long_office_household"],
    "euroc": ["data/euroc/euroc_v1_01_easy",
              "data/euroc/euroc_mh_01_easy"],
}

# The code mirror: everything tracked-ish except the local-only and
# regenerable trees. Excluded receiver-side dirs are also protected
# from --delete by rsync's exclude semantics, so remote output/data
# survive re-syncs.
SYNC_EXCLUDES = [
    ".venv", ".git", "output", "data", "videos", "weights",
    ".pytest_cache", "__pycache__", ".DS_Store", "debug.txt",
    "*.pyc", ".vscode", "*.egg-info",
]


def _ssh(host, args, timeout=None):
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           host] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                         timeout=timeout)


def _rsync(source, dest, host, extra=()):
    # macOS ships openrsync (2.6.9-compatible): no --info= levels,
    # so keep to flags that protocol 29 supports. Plain -a, quiet on
    # success; callers surface failures.
    cmd = ["rsync", "-a",
           "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10"]
    cmd += extra
    cmd += [source, f"{host}:{dest}"]
    return subprocess.run(cmd, capture_output=True, text=True)


def cmd_probe(args):
    r = _ssh(args.host, [
        "uname -srmo; nproc; free -h | sed -n 2p; "
        "df -h --output=size,avail / | sed -n 2p; "
        "uv --version; python3 --version; "
        f"test -d {REMOTE_DIR} && echo MIRROR:present || echo MIRROR:absent; "
        f"test -x {REMOTE_DIR}/.venv/bin/python && "
        "echo VENV:present || echo VENV:absent"
    ])
    print(r.stdout or r.stderr)
    if r.returncode != 0:
        print("[remote] probe FAILED (is key-based ssh set up?)")
        return 1
    return 0


def cmd_sync(args):
    t0 = time.time()
    excludes = sum((["--exclude", e] for e in SYNC_EXCLUDES), [])
    r = _rsync(f"{REPO_ROOT}/", REMOTE_DIR + "/", args.host, excludes)
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return 1
    print(f"[remote] code synced in {time.time() - t0:.1f}s")

    def _sync_tree(local, label):
        p = REPO_ROOT / local
        if not p.exists():
            print(f"[remote] nothing at {local}; skipping {label}")
            return
        t = time.time()
        rr = _rsync(f"{p}/", f"{REMOTE_DIR}/{local}/", args.host)
        print(f"[remote] {label} synced in {time.time() - t0:.1f}s"
              if rr.returncode == 0 else rr.stderr)

    if args.videos or args.all_videos:
        if args.all_videos:
            _sync_tree("videos", "videos (all)")
        else:
            for v in args.videos:
                rr = _rsync(f"{REPO_ROOT / 'videos' / v}",
                            f"{REMOTE_DIR}/videos/", args.host)
                print(f"[remote] video {v}: "
                      f"{'ok' if rr.returncode == 0 else rr.stderr}")
    for d in args.data or []:
        for sub in DATA_TREES[d]:
            # openrsync does not create intermediate parent dirs on
            # the receiver — pre-create them (idempotent).
            _ssh(args.host, [f"mkdir -p {REMOTE_DIR}/"
                             f"{'/'.join(sub.split('/')[:-1])}"])
            t = time.time()
            rr = _rsync(f"{REPO_ROOT / sub}/", f"{REMOTE_DIR}/{sub}/",
                        args.host, extra=["-L"])
            status = ("ok" if rr.returncode == 0
                      else (rr.stderr or rr.stdout))
            print(f"[remote] {sub}: {status} "
                  f"({time.time() - t:.0f}s)", flush=True)
    if args.weights:
        _sync_tree("weights", "weights")

    # env provision (idempotent; installs the pinned python per
    # .python-version + uv.lock on first run)
    print("[remote] uv sync (env provision) ...", flush=True)
    # --all-extras: the gate machine holds the superset (calib +
        # learned) - phases measure every optional stack here.
        r = _ssh(args.host, [f"cd {REMOTE_DIR} && uv sync --frozen --all-extras"])
    print(r.stdout[-2000:] if r.stdout else "", end="")
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return 1
    print("[remote] env ready")
    return 0


def cmd_run(args):
    if not args.command:
        print("[remote] run needs -- COMMAND (after the -- separator)")
        return 1
    started = time.strftime("%Y%m%d_%H%M%S")
    r = _ssh(args.host, ["uname -srmo; nproc; hostname"])
    platform = (r.stdout or "").strip().replace("\n", " ")
    print(f"[remote] host: {args.host} | {platform}")
    print(f"[remote] command: {' '.join(args.command)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         f"cd {REMOTE_DIR} && uv run " + " ".join(args.command)])
    elapsed = time.time() - t0
    print(f"[remote] exit {proc.returncode} after {elapsed:.0f}s")
    if proc.returncode != 0:
        return proc.returncode
    # provenance manifest (fetched artifacts are identified by it)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    slug = "_".join(args.command).replace("/", "-")[:60]
    manifest = MANIFEST_DIR / f"{started}_{slug}.json"
    manifest.write_text(json.dumps({
        "host": args.host, "remote_dir": REMOTE_DIR,
        "platform": platform, "command": args.command,
        "started": started,
        "finished": time.strftime("%Y%m%d_%H%M%S"),
        "elapsed_seconds": elapsed,
        "exit_code": proc.returncode,
        "note": "produced on a remote machine: cross-machine float "
                "paths differ; compare A/B legs within ONE machine "
                "only. Segment-level diagnostics are the cross-"
                "machine currency.",
    }, indent=1) + "\n")
    print(f"[remote] manifest: {manifest}")
    print("[remote] fetch when ready: python -m benchmarks.remote fetch")
    return 0


def cmd_fetch(args):
    extra = [] if args.force else ["--ignore-existing"]
    r = subprocess.run(
        ["rsync", "-a",
         "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10"] + extra +
        [f"{args.host}:{REMOTE_DIR}/output/",
         str(REPO_ROOT / "output" / "")], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return 1
    note = ("--force: collisions OVERWRITTEN (explicit)"
            if args.force else
            "collisions SKIPPED (--ignore-existing; local evidence "
            "is never silently overwritten)")
    print(f"[remote] fetch done. {note}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe")

    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--videos", nargs="*", default=None,
                        metavar="FILE.mp4")
    p_sync.add_argument("--all-videos", action="store_true")
    p_sync.add_argument("--data", nargs="*", default=None,
                        choices=["tum", "euroc"],
                        help="sync the NORMALIZED sequence trees for "
                             "remote ladder runs (symlinks are "
                             "dereferenced; raw/bundle data stays "
                             "local)")
    p_sync.add_argument("--weights", action="store_true")

    p_run = sub.add_parser("run")
    p_run.add_argument("command", nargs=argparse.REMAINDER)

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--force", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "run" and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return {"probe": cmd_probe, "sync": cmd_sync,
            "run": cmd_run, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
