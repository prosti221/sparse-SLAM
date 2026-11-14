"""6e read-only workload observer. Nest OUTSIDE KeyframeAudit.

Counts are object/graph sizes, NOT process RSS or a total memory bound.
Sample monotonic map mutations at entry/exit and local/fresh solver graphs
before ephemeral handles or persistent-cache invalidations hide their size.
Fresh graph handles live only until their original solve wrapper returns;
no optimizer calls, state refresh, RNG, quality reads or registry repair.
All timings include diagnostics and are not production speed measurements.
"""
from contextlib import ExitStack
from time import perf_counter
from unittest.mock import patch

import numpy as np

from benchmarks.keyframe_audit import distribution
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
from slam.core.map import Map
from slam.core.tracker import Tracker
from slam.utils import constants as const

_active = False


class KeyframeCostAudit:
    def __init__(self, source_indices, timestamps=None):
        self.source_indices = np.asarray(source_indices, dtype=int)
        self.timestamps = None if timestamps is None else np.asarray(timestamps, dtype=float)
        if self.timestamps is not None and len(self.timestamps) != len(self.source_indices):
            raise ValueError("Timestamp and source-index populations differ")
        self.frames = []
        self.map_samples = []
        self.graphs = []
        self.loop_windows = []
        self._indices = {}
        self._index = -1
        self._fresh = []
        self._last_correction = None
        self.collection_seconds = 0.0

    def window(self, keyframes):
        indices = [self._indices[kf.frame_id] for kf in keyframes if kf.frame_id in self._indices]
        source = [int(self.source_indices[i]) for i in indices]
        times = [float(self.timestamps[i]) for i in indices] if self.timestamps is not None else None
        return {"input_indices": indices, "source_indices": source,
                "source_frame_span": max(source) - min(source) if source else None,
                "acquisition_seconds_span": max(times) - min(times) if times else None,
                "estimated_path_map_units": float(sum(np.linalg.norm(b.pose[:3, 3] - a.pose[:3, 3])
                                                        for a, b in zip(keyframes, keyframes[1:])))}

    def _map(self, world, stage):
        start = perf_counter()
        row = {"input_index": self._index, "stage": stage,
               "points": len(world.points), "keyframes": len(world.keyframes),
               "point_side_observations": sum(len(p.observations) for p in world.points),
               "loop_db_descriptors": sum(len(d) for d in world.feature_db.feature_storage.values()),
               "loop_db_descriptor_bytes": sum(d.nbytes for d in world.feature_db.feature_storage.values())}
        self.map_samples.append(row)
        self.collection_seconds += perf_counter() - start
        return row

    def _graph(self, opt, poses, kind, window=None):
        if opt is None:
            return
        start = perf_counter()
        vertices, edges = len(opt.vertices()), len(opt.edges())
        row = {"input_index": self._index, "kind": kind, "vertices": vertices,
               "pose_vertices": len(poses), "point_vertices": vertices - len(poses),
               "fixed_pose_vertices": sum(bool(v.fixed()) for v in poses.values()), "edges": edges}
        if window is not None:
            row["window"] = window
        self.graphs.append(row)
        self.collection_seconds += perf_counter() - start

    def __enter__(self):
        global _active
        if _active:
            raise RuntimeError("KeyframeCostAudit contexts cannot overlap")
        _active = True
        self._stack = ExitStack()
        original_update = Tracker.update
        original_loop = Tracker._handle_loop_closure_detection
        original_local = G2OBundleAdjustment._reconcile_and_optimize
        original_vertices = G2OBundleAdjustment._create_keyframe_vertices

        def update(tracker, frame):
            self._index += 1
            if self._index >= len(self.source_indices):
                raise RuntimeError("More frames than the declared input manifest")
            self._indices[frame.frame_id] = self._index
            result = original_update(tracker, frame)
            row = self._map(tracker.map, "frame_end")
            self.frames.append({**row, "source_index": int(self.source_indices[self._index]),
                                "features": len(frame.keypoints) if frame.keypoints is not None else 0,
                                "retained_keyframe": frame.frame_id in tracker.map.keyframes_by_id})
            return result

        def local(optimizer, local_keyframes, *args, **kwargs):
            start = perf_counter()
            window = self.window(local_keyframes)
            self.collection_seconds += perf_counter() - start
            try:
                return original_local(optimizer, local_keyframes, *args, **kwargs)
            finally:
                native = optimizer._native_local_ba
                row = {"input_index": self._index, "kind": "local_ba",
                       "vertices": native.num_frames() + native.num_points(),
                       "pose_vertices": native.num_frames(),
                       "point_vertices": native.num_points(),
                       "fixed_pose_vertices": native.fixed_pose_count(),
                       "edges": native.num_edges()}
                if window is not None:
                    row["window"] = window
                self.graphs.append(row)

        def vertices(optimizer, opt, *args, **kwargs):
            result = original_vertices(optimizer, opt, *args, **kwargs)
            if self._fresh:
                self._fresh[-1]["opt"] = opt
                self._fresh[-1]["poses"] = result
            return result

        def fresh_wrapper(original, kind):
            def solve(optimizer, *args, **kwargs):
                record = {"opt": None, "poses": {}}
                self._fresh.append(record)
                try:
                    return original(optimizer, *args, **kwargs)
                finally:
                    if record["opt"] is not None:
                        self._graph(record["opt"], record["poses"], kind)
                    elif kind == "fresh_ba":
                        # The native global solve has no g2o objects
                        # to sample; the same row shape, from the
                        # solve's own inputs.
                        row = getattr(optimizer, "_last_global_graph", None)
                        if row:
                            self.graphs.append({"input_index": self._index,
                                                "kind": kind, **row})
                    self._fresh.pop()  # Do not retain a solver handle in archived data.
            return solve

        def loop(tracker):
            start = perf_counter()
            ids = tracker.map.feature_db.keyframe_order[-const.LOOP_EXCLUDE_RECENT_KF:]
            keyframes = [tracker.map.keyframes_by_id[fid] for fid in ids
                         if fid in tracker.map.keyframes_by_id]
            row = {"input_index": self._index, "excluded_recent": self.window(keyframes),
                   "cooldown_counter_before": tracker._loop_kfs_since_correction,
                   "last_correction_input_index": self._last_correction,
                   "source_frames_since_correction": (int(self.source_indices[self._index]
                                                          - self.source_indices[self._last_correction])
                                                       if self._last_correction is not None else None)}
            self.collection_seconds += perf_counter() - start
            result = original_loop(tracker)
            # This flag is set by a successful correction, then consumed by
            # update AFTER this hook. Observe it without introducing telemetry.
            row["corrected"] = bool(tracker._loop_correction_applied)
            if row["corrected"]:
                self._last_correction = self._index
            self.loop_windows.append(row)
            return result

        def map_wrapper(original, name):
            def mutation(world, *args, **kwargs):
                self._map(world, name + "_before")
                result = original(world, *args, **kwargs)
                self._map(world, name + "_after")
                return result
            return mutation

        bindings = [(Tracker, "update", update), (Tracker, "_handle_loop_closure_detection", loop),
                    (G2OBundleAdjustment, "_reconcile_and_optimize", local),
                    (G2OBundleAdjustment, "_create_keyframe_vertices", vertices)]
        for name, kind in (("_optimize_with_g2o", "fresh_ba"), ("pose_graph_optimization", "pose_graph")):
            bindings.append((G2OBundleAdjustment, name,
                             fresh_wrapper(getattr(G2OBundleAdjustment, name), kind)))
        for name in ("add_keyframe", "add_points", "prune_outlier_points", "_remove_keyframe", "reset"):
            bindings.append((Map, name, map_wrapper(getattr(Map, name), name)))
        for owner, name, wrapper in bindings:
            self._stack.enter_context(patch.object(owner, name, wrapper))
        return self

    def __exit__(self, *exc):
        global _active
        try:
            return self._stack.__exit__(*exc)
        finally:
            _active = False

    def summary(self, telemetry):
        n = len(self.frames)
        peaks = {key: max((row[key] for row in self.map_samples), default=0)
                 for key in ("points", "keyframes", "point_side_observations",
                             "loop_db_descriptors", "loop_db_descriptor_bytes")}
        graphs = {}
        for kind in ("local_ba", "fresh_ba", "pose_graph"):
            rows = [row for row in self.graphs if row["kind"] == kind]
            graphs[kind] = {"samples": len(rows), **{
                "peak_" + key: max((row[key] for row in rows), default=0)
                for key in ("vertices", "pose_vertices", "point_vertices", "fixed_pose_vertices", "edges")}}
        local = [row["window"] for row in self.graphs if row["kind"] == "local_ba" and "window" in row]
        return {"input_frames": n, "map_peaks": peaks, "graph_peaks": graphs,
                "feature_count": distribution(row["features"] for row in self.frames),
                "local_window_source_frame_span": distribution(row["source_frame_span"] for row in local),
                "local_window_acquisition_seconds_span": distribution(row["acquisition_seconds_span"] for row in local),
                "loop_exclusion_source_frame_span": distribution(
                    row["excluded_recent"]["source_frame_span"] for row in self.loop_windows),
                "stage_calls_per_input": {k: v / max(n, 1) for k, v in telemetry["stage_calls"].items()},
                "events_per_input": {k: v / max(n, 1) for k, v in telemetry["events"].items()},
                "diagnostic_collection_seconds": self.collection_seconds,
                "warning": "Per-graph/object-count peaks, not combined process memory or production speed. Includes non-KF point observers; graphs use only KF constraints."}
