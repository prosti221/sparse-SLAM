import g2o
import numpy as np
from slam.core.map import Map
from slam.core.frame import Frame
from slam.core.point import Point
from typing import List, Tuple
from slam.utils.logger import *
from slam.utils.constants import *
from uuid import UUID
from typing import List, Optional, Tuple, List, Dict

LOG_TAG = 'g2oBA'


class G2OBundleAdjustment:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.map = None

    def local_bundle_adjustment(self, map_obj: Map, reference_frame_id: UUID, window_size: int = 5) -> bool:
        self.map = map_obj
        debug_log(
            LOG_TAG, f"Starting local BA with g2o around keyframe {reference_frame_id}")

        local_keyframes = map_obj.get_local_keyframes(
            reference_frame_id, window_size)
        local_points = map_obj.get_local_points(
            local_keyframes, min_observations=2)

        if len(local_keyframes) < MINIMUM_LOCAL_KEYFRAMES or len(local_points) < MINIMUM_LOCAL_POINTS:
            warning_log(
                LOG_TAG, f"Insufficient data for BA: {len(local_keyframes)} keyframes, {len(local_points)} points")
            return False

        observations = map_obj.get_observations(
            local_keyframes, local_points, normalize_points=True)

        if len(observations) < MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT:
            warning_log(
                LOG_TAG, f"Insufficient observations for BA: {len(observations)}")
            return False

        return self._optimize_with_g2o(local_keyframes, local_points, observations, fix_points=False)

    def global_bundle_adjustment(self, map_obj: Map, max_keyframes: Optional[int] = None) -> bool:
        self.map = map_obj
        keyframes = map_obj.keyframes
        points = map_obj.points
        observations = map_obj.get_observations(
            keyframes, points, normalize_points=True)

        if len(observations) < MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT:
            warning_log(
                LOG_TAG, f"Insufficient observations for global BA: {len(observations)}")
            return False

        return self._optimize_with_g2o(keyframes, points, observations, fix_points=False)

    def _optimize_with_g2o(self, keyframes: List[Frame], points: List[Point],
                           observations: List[Tuple[int, int, np.ndarray]],
                           fix_points: bool = False) -> bool:
        fix_points = True
        # create g2o optimizer
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)

        # add normalized camera
        cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
        cam.set_id(0)
        opt.add_parameter(cam)

        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
        graph_frames, graph_points = {}, {}

        # add frames to graph
        for i, f in enumerate(keyframes):
            pose = f.pose
            se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_estimate(se3)

            v_se3.set_id(i * 2)
            v_se3.set_fixed(i <= 0)
            opt.add_vertex(v_se3)

            # confirm pose correctness
            est = v_se3.estimate()
            assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
            assert np.allclose(pose[0:3, 3], est.translation())

            graph_frames[f.frame_id] = v_se3

        # add points to frames
        for i, p in enumerate(points):
            if not any([f in [kf.frame_id for kf in keyframes] for f in p.observations]):
                continue

            pt = g2o.VertexPointXYZ()
            pt.set_id(i * 2 + 1)
            pt.set_estimate(p.pt_3d[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(fix_points)
            opt.add_vertex(pt)
            graph_points[p.point_id] = pt

            # add edges
            for frame_id, (keypoint_idx, pt_2d) in p.observations.items():
                if frame_id not in graph_frames:
                    continue
                f = self.map.get_keyframe_by_id(frame_id)
                keypoint = f.keypoints[keypoint_idx]
                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_parameter_id(0, 0)
                edge.set_vertex(0, pt)
                edge.set_vertex(1, graph_frames[frame_id])
                edge.set_measurement(
                    f.normalize_keypoint(np.array([keypoint.pt[0], keypoint.pt[1]])))
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)

        rounds = 40
        opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(rounds)

        # put frames back
        for f in graph_frames:
            frame = self.map.get_keyframe_by_id(f)
            est = graph_frames[f].estimate()
            R = est.rotation().matrix()
            t = est.translation()
            frame.pose = self.poseRt(R, t)

        # put points back
        if not fix_points:
            for p in graph_points:
                point = self.map.get_point_by_id(p)
                point.pt_3d = np.array(graph_points[p].estimate())

        return opt.active_chi2()

    def poseRt(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[0:3, 0:3] = R
        pose[0:3, 3] = t
        return pose
