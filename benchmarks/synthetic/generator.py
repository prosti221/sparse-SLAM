"""Deterministic synthetic world generator for SLAM benchmarking.

Generates fully reproducible sequences: a textured box room rendered from a
known camera trajectory. Because billboards are rigid world-space quads
(perspective-warped into each frame), every visual feature corresponds to a
fixed 3D point - making both trajectory AND map evaluation meaningful.

Output layout (per sequence, under <repo>/data/synthetic/<name>/):
    frames/frame_%06d.png   BGR frames
    trajectory.txt          TUM format: "i tx ty tz qx qy qz qw" (Twc poses)
    calibration.txt         "fx fy cx cy"
    scene_points.txt        billboard centers (x y z), one per line
    scene_planes.txt        wall planes "nx ny nz d" (inward normals, n.p = d)
    meta.json               generation parameters
"""
import json
from pathlib import Path
from typing import Callable, List, Tuple

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

# --- Fixed parameters (identical across sequences -> comparable results) ---
WORLD_SEED = 42
IMG_W, IMG_H = 640, 480
FX = FY = 450.0
CX, CY = IMG_W / 2.0, IMG_H / 2.0
PATTERN_SIZE = 32          # texture resolution per billboard (px)
BILLBOARD_SIZE = 0.7       # world-space extent (square)
PATCHES_PER_WALL = 200
WALL_MARGIN = 0.6           # keep billboards away from wall edges

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "synthetic"

# Wall planes: (inward normal n, plane offset d with n.p = d,
#              basis origin, tangent u, tangent v, u range, v range)
_WallSpec = List[Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, tuple, tuple]]
WALLS: _WallSpec = [
    # x = +10
    (np.array([-1.0, 0.0, 0.0]), -10.0,
     np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]),
     (0.0 + WALL_MARGIN, 12.0 - WALL_MARGIN), (-6.0 + WALL_MARGIN, 6.0 - WALL_MARGIN)),
    # x = -10
    (np.array([1.0, 0.0, 0.0]), -10.0,
     np.array([-10.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]),
     (0.0 + WALL_MARGIN, 12.0 - WALL_MARGIN), (-6.0 + WALL_MARGIN, 6.0 - WALL_MARGIN)),
    # y = +6 (ceiling)
    (np.array([0.0, -1.0, 0.0]), -6.0,
     np.array([0.0, 6.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
     (-10.0 + WALL_MARGIN, 10.0 - WALL_MARGIN), (0.0 + WALL_MARGIN, 12.0 - WALL_MARGIN)),
    # y = -6 (floor)
    (np.array([0.0, 1.0, 0.0]), -6.0,
     np.array([0.0, -6.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
     (-10.0 + WALL_MARGIN, 10.0 - WALL_MARGIN), (0.0 + WALL_MARGIN, 12.0 - WALL_MARGIN)),
    # z = 12 (far wall)
    (np.array([0.0, 0.0, -1.0]), -12.0,
     np.array([0.0, 0.0, 12.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
     (-10.0 + WALL_MARGIN, 10.0 - WALL_MARGIN), (-6.0 + WALL_MARGIN, 6.0 - WALL_MARGIN)),
    # z = 0 (back wall)
    (np.array([0.0, 0.0, 1.0]), 0.0,
     np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
     (-10.0 + WALL_MARGIN, 10.0 - WALL_MARGIN), (-6.0 + WALL_MARGIN, 6.0 - WALL_MARGIN)),
]

# --- Trajectories (all deterministic, no RNG) -----------------------------

def _translation_pose(i: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pure lateral motion: strong parallax, constant orientation."""
    t = np.array([-6.0 + 0.1 * i, 0.0, 0.5])
    return np.eye(3), t


def _orbit_pose(i: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Circle around the room center, always looking radially outward."""
    theta = 2.0 * np.pi * i / n
    center = np.array([0.0, 0.0, 4.0])
    t = center + np.array([3.0 * np.cos(theta), 0.0, 3.0 * np.sin(theta)])
    forward = t - center
    yaw = np.arctan2(forward[0], forward[2])
    R = Rotation.from_euler("y", yaw).as_matrix()
    return R, t


def _forward_rotation_pose(i: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Forward motion with oscillating yaw: mixed translation + rotation."""
    t = np.array([0.0, 0.0, 1.0 + 0.08 * i])
    yaw = 0.35 * np.sin(2.0 * np.pi * i / 100)
    R = Rotation.from_euler("y", yaw).as_matrix()
    return R, t


SEQUENCES = {
    "translation": {"pose_fn": _translation_pose, "n_frames": 120,
                    "description": "Pure lateral translation, constant orientation"},
    "orbit": {"pose_fn": _orbit_pose, "n_frames": 150,
              "description": "Circular path, radially outward-facing camera"},
    "forward_rotation": {"pose_fn": _forward_rotation_pose, "n_frames": 120,
                          "description": "Forward motion with oscillating yaw"},
}

# --- World construction ----------------------------------------------------

def _make_pattern(rng: np.random.Generator) -> np.ndarray:
    """A high-contrast random texture: corner-rich, unique per billboard."""
    pat = rng.integers(85, 115, size=(PATTERN_SIZE, PATTERN_SIZE), dtype=np.uint8)
    speckle_mask = rng.random(pat.shape) < 0.45
    vals = rng.choice(np.array([15, 240], dtype=np.uint8), size=pat.shape)
    pat[speckle_mask] = vals[speckle_mask]
    return pat


def _build_world() -> List[dict]:
    """Billboards: {center, u, v, corners_world, pattern_bgr}."""
    rng = np.random.default_rng(WORLD_SEED)
    billboards = []
    half = BILLBOARD_SIZE / 2.0
    for (n, d, origin, u, v, urange, vrange) in WALLS:
        for _ in range(PATCHES_PER_WALL):
            s_u = rng.uniform(*urange)
            s_v = rng.uniform(*vrange)
            center = origin + u * s_u + v * s_v
            corners = np.array([
                center - u * half - v * half,   # texture top-left
                center + u * half - v * half,   # texture top-right
                center + u * half + v * half,   # texture bottom-right
                center - u * half + v * half,   # texture bottom-left
            ])
            gray = _make_pattern(rng)
            pattern_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            billboards.append({
                "center": center,
                "corners": corners,
                "pattern": pattern_bgr,
            })
    return billboards


def _project(points: np.ndarray, R_cw: np.ndarray, t_cw: np.ndarray,
             K: np.ndarray) -> np.ndarray:
    """World (N,3) -> pixel (N,2); assumes all z > 0."""
    cam = (R_cw @ points.T).T + t_cw
    z = cam[:, 2:3]
    uv = (K @ (cam / z).T).T
    return uv[:, :2]


def _paste_billboard(img: np.ndarray, pattern: np.ndarray, quad: np.ndarray) -> None:
    """Perspective-warp a billboard texture into the image at its projected quad."""
    h, w = img.shape[:2]
    x0 = max(int(np.floor(quad[:, 0].min())) - 1, 0)
    y0 = max(int(np.floor(quad[:, 1].min())) - 1, 0)
    x1 = min(int(np.ceil(quad[:, 0].max())) + 1, w)
    y1 = min(int(np.ceil(quad[:, 1].max())) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return

    src = np.float32([[0, 0], [PATTERN_SIZE, 0],
                      [PATTERN_SIZE, PATTERN_SIZE], [0, PATTERN_SIZE]])
    dst = np.float32(quad - np.array([x0, y0]))
    M = cv.getPerspectiveTransform(src, dst)

    sub_w, sub_h = x1 - x0, y1 - y0
    warped = cv.warpPerspective(pattern, M, (sub_w, sub_h))
    mask = cv.warpPerspective(
        np.full((PATTERN_SIZE, PATTERN_SIZE), 255, np.uint8), M, (sub_w, sub_h))
    roi = img[y0:y1, x0:x1]
    roi[mask > 0] = warped[mask > 0]


def _render_frame(billboards: List[dict], R_wc: np.ndarray, t_wc: np.ndarray,
                  K: np.ndarray) -> np.ndarray:
    img = np.full((IMG_H, IMG_W, 3), 40, np.uint8)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    margin = 80

    for bb in billboards:
        corners_cam = (R_cw @ bb["corners"].T).T + t_cw
        if np.any(corners_cam[:, 2] <= 0.2):
            continue  # partially behind camera
        center_cam = R_cw @ bb["center"] + t_cw
        if not (-margin < center_cam[0] / center_cam[2] * FX + CX < IMG_W + margin
                and -margin < center_cam[1] / center_cam[2] * FY + CY < IMG_H + margin):
            continue  # center far out of view

        quad = _project(bb["corners"], R_cw, t_cw, K)
        if (quad[:, 0].max() - quad[:, 0].min()) < 5 \
                or (quad[:, 1].max() - quad[:, 1].min()) < 5:
            continue  # sliver / too small
        _paste_billboard(img, bb["pattern"], quad)

    return img


def generate_sequence(name: str, force: bool = False) -> Path:
    """Generate (or reuse) a benchmark sequence. Returns its directory."""
    if name not in SEQUENCES:
        raise ValueError(f"Unknown sequence '{name}'. Available: {sorted(SEQUENCES)}")
    spec = SEQUENCES[name]

    out_dir = DATA_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_file = out_dir / "trajectory.txt"
    if traj_file.exists() and not force:
        return out_dir

    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    billboards = _build_world()

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    n = spec["n_frames"]
    with open(traj_file, "w") as traj:
        for i in range(n):
            R, t = spec["pose_fn"](i, n)
            img = _render_frame(billboards, R, t, K)
            cv.imwrite(str(frames_dir / f"frame_{i:06d}.png"), img)

            quat = Rotation.from_matrix(R).as_quat()  # x y z w
            traj.write(f"{i} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                       f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}\n")

    with open(out_dir / "calibration.txt", "w") as f:
        f.write(f"{FX} {FY} {CX} {CY}\n")

    with open(out_dir / "scene_points.txt", "w") as f:
        for bb in billboards:
            c = bb["center"]
            f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

    with open(out_dir / "scene_planes.txt", "w") as f:
        for (nvec, d, *_rest) in WALLS:
            f.write(f"{nvec[0]} {nvec[1]} {nvec[2]} {d}\n")

    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "name": name,
            "seed": WORLD_SEED,
            "n_frames": n,
            "image_size": [IMG_W, IMG_H],
            "billboard_size": BILLBOARD_SIZE,
            "patches_per_wall": PATCHES_PER_WALL,
            "description": spec["description"],
        }, f, indent=2)

    print(f"[generator] '{name}': {n} frames, {len(billboards)} billboards -> {out_dir}")
    return out_dir


def ensure_sequence(name: str) -> Path:
    """Generate the sequence if it doesn't exist yet, then return its path."""
    return generate_sequence(name, force=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", default=sorted(SEQUENCES))
    parser.add_argument("--force", action="store_true", help="regenerate even if present")
    args = parser.parse_args()
    for seq in args.sequences:
        generate_sequence(seq, force=args.force)
