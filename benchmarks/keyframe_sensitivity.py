"""6e fixed-control sensitivity/data helpers. Never modify canonical inputs.

Half-resolution uses OpenCV resize's pixel-center transform, fixed native
pixel gates, and unchanged feature/BA/loop settings. Stride two changes
input sampling, NOT execution/playback speed. Synthetic timestamps are
unknown; real RGB timestamps come from source filenames/explicit manifests.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import cv2 as cv
import numpy as np

from benchmarks.harness import load_calibration
from benchmarks import metrics
from benchmarks import tum

MODES = ("native", "half_resolution", "stride2")
HELDOUT_NAME = "tum_rpy"
HELDOUT_DATASET = "rgbd_dataset_freiburg1_rpy"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_tree_sha256(directory):
    digest = hashlib.sha256()
    for path in sorted((Path(directory) / "frames").glob("*.png")):
        digest.update(path.name.encode() + b"\0" + file_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def resize_intrinsics(K, sx, sy):
    transform = np.array([[sx, 0., (sx - 1.) / 2], [0., sy, (sy - 1.) / 2], [0., 0., 1.]])
    return transform @ K


def input_timestamps(source, paths):
    manifest = Path(source) / "source_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["timestamps_seconds"]
    # The original TUM adapter preserves RGB timestamp filenames in targets.
    if all(p.is_symlink() for p in paths):
        try:
            return [float(p.resolve().stem) for p in paths]
        except ValueError:
            pass
    return None


def prepare_view(source, destination, mode):
    if mode not in MODES:
        raise ValueError(mode)
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError("Do not overwrite an existing sensitivity dataset")
    paths = sorted((source / "frames").glob("*.png"))
    gt = np.loadtxt(source / "trajectory.txt", ndmin=2)
    if len(paths) != len(gt) or not paths:
        raise ValueError("Frames and GT must have the same nonempty population")
    indices = list(range(0, len(paths), 2 if mode == "stride2" else 1))
    timestamps = input_timestamps(source, paths)
    destination.mkdir(parents=True)
    (destination / "frames").mkdir()
    K, k1 = load_calibration(source)
    first = cv.imread(str(paths[0]))
    if first is None:
        raise IOError(paths[0])
    h, w = first.shape[:2]
    size = (w // 2, h // 2) if mode == "half_resolution" else (w, h)
    for i, original in enumerate(indices):
        target = destination / "frames" / f"frame_{i:06d}.png"
        if mode == "half_resolution":
            image = cv.imread(str(paths[original]))
            if image is None or image.shape[:2] != (h, w):
                raise ValueError(f"Missing or varying-size image: {paths[original]}")
            if not cv.imwrite(str(target), cv.resize(image, size, interpolation=cv.INTER_AREA)):
                raise IOError(target)
        else:
            target.symlink_to(paths[original])
    if mode == "half_resolution":
        K = resize_intrinsics(K, size[0] / w, size[1] / h)
        values = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        if k1 != 0.:
            values.append(k1)
        (destination / "calibration.txt").write_text(" ".join(format(v, ".17g") for v in values) + "\n")
    else:
        (destination / "calibration.txt").symlink_to(source / "calibration.txt")
    if mode == "stride2":
        np.savetxt(destination / "trajectory.txt", gt[indices], fmt="%.17g")
    else:
        (destination / "trajectory.txt").symlink_to(source / "trajectory.txt")
    for filename in ("scene_points.txt", "scene_planes.txt"):
        if (source / filename).exists():
            (destination / filename).symlink_to(source / filename)
    record = {"source_directory": str(source), "mode": mode, "source_frame_count": len(paths),
              "source_indices": indices, "source_image_size": [w, h], "image_size": list(size),
              "timestamps_seconds": [timestamps[i] for i in indices] if timestamps is not None else None,
              "timestamp_units": "acquisition_seconds" if timestamps is not None else "unavailable",
              "source_frame_paths": [str(paths[i].resolve()) for i in indices],
              "source_trajectory_sha256": hashlib.sha256((source / "trajectory.txt").read_bytes()).hexdigest(),
              "calibration_K": K.tolist(), "k1": k1,
              "input_frame_tree_sha256": frame_tree_sha256(destination),
              "warning": "No thresholds rescaled; cadence changes physical sampling. Last sampled frame may precede source end."}
    (destination / "source_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def associate_rgb_gt(raw):
    """Same nearest-GT and later-tie semantics as the accepted XYZ adapter."""
    raw = Path(raw)
    rgb = sorted(tum._read_stamped_list(raw / "rgb.txt"))
    gt = sorted(tum._read_stamped_list(raw / "groundtruth.txt"))
    if not gt:
        raise ValueError("No ground truth")
    result = []
    j = 0
    for timestamp, fields in rgb:
        while j + 1 < len(gt) and abs(gt[j + 1][0] - timestamp) <= abs(gt[j][0] - timestamp):
            j += 1
        gt_time, pose = gt[j]
        relative = PurePosixPath(fields[0])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe RGB path")
        if abs(gt_time - timestamp) <= tum.MAX_TIMESTAMP_DIFF and len(pose) >= 7 and (raw / relative).is_file():
            result.append((timestamp, gt_time, raw / relative, pose[:7]))
    return result


def extract_heldout(archive, root):
    """Extract only regular RGB/metadata files; reject traversal and links."""
    root = Path(root).resolve()
    with tarfile.open(archive) as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive path")
            if not path.parts or path.parts[0] != HELDOUT_DATASET:
                raise ValueError("Unexpected archive root")
            relative = path.parts[1:]
            wanted = relative and (relative[0] == "rgb" or "/".join(relative) in
                                    ("rgb.txt", "groundtruth.txt", "README", "accelerometer.txt"))
            if not wanted or member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Archive links are not accepted")
            target = root.joinpath(*path.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError("Archive target escapes root")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream:
                content = stream.read()
            if target.exists():
                if target.read_bytes() != content:
                    raise FileExistsError(target)
            else:
                target.write_bytes(content)


def matched_sample_comparison(result, native, source_indices, source_gt, timestamps=None):
    """ATE on identical samples; RPE on identical source-frame endpoints.

    Each side has its own Sim(3) fit on that SAME sample population.
    Standard run metrics still describe all of that run's processed frames.
    """
    indices = np.asarray(source_indices, dtype=int)
    gt = source_gt[indices]
    candidate, control = result.poses, native["poses"][indices]
    output = {"common_input_frames": len(indices), "rpe_source_frame_horizon": 10,
              "last_source_frame": int(indices[-1]), "rpe": {}}
    positions = {int(original): i for i, original in enumerate(indices)}
    pairs = [(positions[i], positions[i + 10]) for i in range(0, int(indices[-1]) - 9, 10)
             if i in positions and i + 10 in positions]
    output["rpe_pair_source_indices"] = [[int(indices[i]), int(indices[j])] for i, j in pairs]
    if timestamps is not None:
        output["rpe_pair_acquisition_seconds"] = [float(timestamps[j] - timestamps[i]) for i, j in pairs]
    for label, poses in (("control", control), ("candidate", candidate)):
        alignment = metrics.ate(poses, gt)
        output[label + "_ate"] = alignment["ate_rmse"]
        trans, rotation = [], []
        scaled = metrics._scaled_poses(poses, alignment["scale"])
        for i, j in pairs:
            error = np.linalg.inv(np.linalg.inv(gt[i]) @ gt[j]) @ (np.linalg.inv(scaled[i]) @ scaled[j])
            trans.append(float(np.linalg.norm(error[:3, 3])))
            rotation.append(float(np.degrees(np.arccos(np.clip((np.trace(error[:3, :3]) - 1.) / 2, -1., 1.)))))
        output["rpe"][label] = {"pairs": len(pairs),
                                  "translation_rmse": float(np.sqrt(np.mean(np.square(trans)))) if pairs else None,
                                  "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rotation)))) if pairs else None}
    common = np.intersect1d(native["keyframe_gt_indices"], indices[np.asarray(result.keyframe_gt_indices, dtype=int)])
    output["common_final_kf_ate"] = {"count": len(common)}
    if len(common) >= 3:
        for label, poses, sample in (("control", native["keyframe_poses"], native["keyframe_gt_indices"]),
                                      ("candidate", result.keyframe_poses, indices[result.keyframe_gt_indices])):
            by_index = dict(zip(sample, poses))
            output["common_final_kf_ate"][label] = metrics.ate(np.array([by_index[i] for i in common]), source_gt[common])["ate_rmse"]
    return output
