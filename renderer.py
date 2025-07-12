import open3d as o3d
import numpy as np
from utils import pt_obj_to_array
from logger import debug_log, info_log, error_log, warning_log

LOG_TAG = 'Renderer'


class Renderer:
    def __init__(self, K, width=1920, height=1080):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.width = width
        self.height = height

        self.K = K

        self.point_cloud = o3d.geometry.PointCloud()

        # {keyframe_id: (pose_geometry, optimization_iterations)}
        self.poses = {}

        self.ctrl = None
        self.cloud_initialized = False
        self.camera_initialized = False

        self.pinhole = o3d.camera.PinholeCameraIntrinsic(
            width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self.camera_parameters = o3d.camera.PinholeCameraParameters()
        self.camera_parameters.intrinsic = self.pinhole

        self.paused = False

    def start(self):
        info_log(LOG_TAG, "Starting Open3D visualizer")
        self.vis.create_window(
            window_name="SLAM", width=self.width, height=self.height)
        self.vis.get_render_option().background_color = [0.0, 0.0, 0.0]
        self.ctrl = self.vis.get_view_control()
        self.ctrl.convert_from_pinhole_camera_parameters(
            self.camera_parameters, allow_arbitrary=True)

        # Register spacebar (ASCII 32) to toggle pause
        self.vis.register_key_callback(32, self._toggle_pause)

    def stop(self):
        info_log(LOG_TAG, "Stopping Open3D visualizer")
        self.vis.destroy_window()

    def is_paused(self):
        return self.paused

    def update_points(self, pts):
        if len(pts) == 0:
            error_log(LOG_TAG, "No points to render")
            return

        pts_array, colors = pt_obj_to_array(pts)

        self.point_cloud.points = o3d.utility.Vector3dVector(pts_array)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

        if not self.cloud_initialized:
            self.vis.add_geometry(self.point_cloud)
            self.cloud_initialized = True

        # Update view
        self.vis.get_render_option().point_size = 1.5
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

        debug_log(LOG_TAG, f"Rendering point cloud with {len(pts)} points")

    def update_poses(self, keyframes):
        if not self.camera_initialized:
            first_kf = keyframes[0]
            self._initialize_camera(first_kf.get_pose().copy())

        for kf in keyframes:
            if kf.frame_id not in self.poses:
                debug_log(LOG_TAG, f"Adding new pose for frame {kf.frame_id}")
                new_pose_geometry = self._construct_pose_geometry(kf)
                self.poses[kf.frame_id] = [
                    new_pose_geometry, kf.optimization_iterations
                ]
                self.vis.add_geometry(new_pose_geometry, False)
            elif kf.is_pose_optimized and kf.optimization_iterations > self.poses[kf.frame_id][1]:
                debug_log(LOG_TAG, f"Updating pose for frame {kf.frame_id}")
                self._update_pose_geometry(kf)
            else:
                continue
        self.vis.poll_events()
        self.vis.update_renderer()

    def _toggle_pause(self, vis):
        self.paused = not self.paused
        info_log(LOG_TAG, "Paused" if self.paused else "Resumed")
        return False

    def _initialize_camera(self, pose):
        debug_log(LOG_TAG, "Initializing camera parameters")
        R = pose[:3, :3]
        pose[:3, 3] += 20 * R[:, 2]
        self.camera_parameters.extrinsic = pose
        self.ctrl.convert_from_pinhole_camera_parameters(
            self.camera_parameters, allow_arbitrary=True)
        self.ctrl.set_constant_z_near(10)
        self.camera_initialized = True

    def _construct_pose_geometry(self, keyframe):
        pose = keyframe.get_pose()
        R, t = pose[:3, :3], pose[:3, 3]
        points, lines = self._draw_camera_object(R, t)

        new_cam = o3d.geometry.LineSet()
        new_cam.points = points
        new_cam.lines = lines

        # Set color to green
        colors = np.zeros((len(lines), 3))
        colors[:, 1] = 1
        new_cam.colors = o3d.utility.Vector3dVector(colors)

        return new_cam

    def _update_pose_geometry(self, keyframe):
        pose = keyframe.get_pose()
        R, t = pose[:3, :3], pose[:3, 3]
        points, lines = self._draw_camera_object(R, t)

        self.poses[keyframe.frame_id][0].points = points
        self.poses[keyframe.frame_id][0].lines = lines
        self.poses[keyframe.frame_id][1] = keyframe.optimization_iterations

        self.vis.update_geometry(self.poses[keyframe.frame_id][0])

    def _draw_camera_object(self, R, t, size=0.8):
        _w, _h, _cx, _cy, _f = self.width, self.height, self.K[0,
                                                               2], self.K[1, 2], self.K[0, 0]
        f = 1
        w = _w/_f
        h = _h/_f
        cx = _cx/_f
        cy = _cy/_f

        offset_cx = cx - w/2.0
        offset_cy = cy - h/2.0

        points = [[0, 0, 0],
                  [offset_cx, offset_cy, f],
                  [-0.5 * w, -0.5 * h, f],
                  [0.5 * w, -0.5 * h, f],
                  [0.5 * w, 0.5 * h, f],
                  [-0.5 * w, 0.5 * h, f],
                  [-0.5 * w, -0.5 * h, f]]

        lines = [[0, 1], [2, 3], [3, 4], [4, 5], [5, 6], [
            0, 2], [0, 3], [0, 4], [0, 5], [2, 4], [3, 5]]

        points = np.array(points) * size

        points = (R @ points.T).T + t

        points = o3d.utility.Vector3dVector(points)
        lines = o3d.utility.Vector2iVector(lines)

        return points, lines
