import open3d as o3d
import numpy as np
import cv2 as cv
import json
import os
from datetime import datetime

from slam.utils.utils import pt_obj_to_array
from slam.utils.logger import debug_log, info_log, error_log, warning_log
from slam.utils.constants import TRACKING_QUALITY_GRADIENT
from slam.core.map import Map

LOG_TAG = 'Renderer'


class Renderer:
    def __init__(self, map: Map, K, W=1920, H=1080):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.W = int(W)
        self.H = int(H)

        self.K = K
        self.map: Map = map

        self.point_cloud = o3d.geometry.PointCloud()

        # {keyframe_id: (pose_geometry, tracking_quality)}
        self.poses = {}

        self.ctrl = None
        self.cloud_initialized = False
        self.camera_initialized = False

        self.pinhole = o3d.camera.PinholeCameraIntrinsic(
            self.W, self.H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self.camera_parameters = o3d.camera.PinholeCameraParameters()
        self.camera_parameters.intrinsic = self.pinhole

        self.paused = False
        self.quit_requested = False  # Add quit flag

        # Camera tracking attributes
        self.tracking_enabled = False
        self.last_tracked_frame = None

        # Save/load attributes
        self.session_data = {
            'points': [],
            'poses': {},
            'camera_params': None,
            'timestamp': None
        }

        # Video recording attributes
        self.recording = False
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = f"output/sessions/{self.timestamp}"
        self.video_writer = None
        self.video_filename = None
        self.frame_count = 0

    def start(self):
        info_log(LOG_TAG, "Starting Open3D visualizer")
        self.vis.create_window(
            window_name="SLAM", width=self.W, height=self.H)
        self.vis.get_render_option().background_color = [0.0, 0.0, 0.0]

        self.ctrl = self.vis.get_view_control()
        self.ctrl.convert_from_pinhole_camera_parameters(
            self.camera_parameters, allow_arbitrary=True)
        self.ctrl.set_constant_z_far(10000.0)

        # Register spacebar (ASCII 32) to toggle pause
        self.vis.register_key_callback(32, self._toggle_pause)

        # Register 'q' key (ASCII 113) to quit
        self.vis.register_key_callback(113, self._request_quit)

        # Register 't' key (ASCII 116) to toggle camera tracking
        self.vis.register_key_callback(84, self._toggle_tracking)

        # Create session output folder
        os.makedirs(self.output_path, exist_ok=True)

    def stop(self):
        info_log(LOG_TAG, "Stopping Open3D visualizer")
        if self.recording:
            self._stop_recording()
        self.vis.destroy_window()

    def is_paused(self):
        return self.paused

    def should_quit(self):
        return self.quit_requested

    def update(self):
        self.update_points(self.map.points)
        self.update_poses(self.map.keyframes)

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

        # Store points for session saving
        self.session_data['points'] = {
            'positions': pts_array.tolist(),
            'colors': colors.tolist()
        }

    def update_poses(self, keyframes):
        # Get current keyframe IDs from the map
        current_keyframe_ids = {kf.frame_id for kf in keyframes}

        # Remove poses that are no longer in the map
        self._remove_deleted_poses(current_keyframe_ids)

        for kf in keyframes:
            if kf.frame_id not in self.poses:
                debug_log(LOG_TAG, f"Adding new pose for frame {kf.frame_id}")
                new_pose_geometry = self._construct_pose_geometry(kf)
                self.poses[kf.frame_id] = [
                    new_pose_geometry, kf.tracking_quality
                ]
                self.vis.add_geometry(new_pose_geometry, False)
            elif self.poses[kf.frame_id][1] != kf.tracking_quality:
                debug_log(LOG_TAG, f"Updating pose for frame {kf.frame_id}")
                self._update_pose_geometry(kf)
            else:
                continue

        # Store pose data for session saving
        self.session_data['poses'][kf.frame_id.int] = {
            'pose': kf.pose.tolist(),
            'tracking_quality': kf.tracking_quality,
        }

        # Update camera tracking if enabled
        self.last_tracked_frame = keyframes[-1]
        if self.tracking_enabled or not self.camera_initialized:
            self.camera_initialized = True
            self._update_camera_tracking()

        self.vis.poll_events()
        self.vis.update_renderer()

    def _toggle_pause(self, vis):
        self.paused = not self.paused
        info_log(LOG_TAG, "Paused" if self.paused else "Resumed")
        return False

    def _toggle_tracking(self, vis):
        self.tracking_enabled = not self.tracking_enabled
        status = "enabled" if self.tracking_enabled else "disabled"
        info_log(LOG_TAG, f"Camera tracking {status}")

        if not self.tracking_enabled:
            # Reset tracking state when disabled
            self.last_tracked_frame = None

        return False

    def _request_quit(self, vis):
        self.quit_requested = True
        info_log(LOG_TAG, "Ending slam session...")
        return False

    def _construct_pose_geometry(self, keyframe):
        pose = keyframe.pose
        R, t = pose[:3, :3], pose[:3, 3]
        points, lines = self._draw_camera_object(R, t)

        new_cam = o3d.geometry.LineSet()
        new_cam.points = points
        new_cam.lines = lines

        # Set current frame color to cyan
        colors = np.tile(np.array([0, 255, 255]), (len(lines), 1))
        new_cam.colors = o3d.utility.Vector3dVector(colors)

        self.last_tracked_frame = keyframe

        return new_cam

    def _update_pose_geometry(self, keyframe):
        pose = keyframe.pose
        R, t = pose[:3, :3], pose[:3, 3]
        points, lines = self._draw_camera_object(R, t)

        self.poses[keyframe.frame_id][0].points = points
        self.poses[keyframe.frame_id][0].lines = lines
        quality_color = self._get_tracking_quality_color(
            keyframe.tracking_quality)
        colors = np.tile(quality_color, (len(lines), 1))
        self.poses[keyframe.frame_id][0].colors = o3d.utility.Vector3dVector(
            colors)
        self.poses[keyframe.frame_id][1] = keyframe.tracking_quality

        self.vis.update_geometry(self.poses[keyframe.frame_id][0])

    def _update_camera_tracking(self):
        # Get the pose of the most recent keyframe
        tracked_pose = np.linalg.inv(self.last_tracked_frame.pose)
        tracked_position = tracked_pose[:3, 3]

        # Project position to ground plane (assuming y is up)
        ground_position = tracked_position.copy()
        ground_position[1] = 0  # Project to ground

        # Fixed top-down camera setup
        camera_height = 20  # Fixed height above ground
        tilt_angle = np.radians(25)

        # Camera position
        # TODO: This way of positioning the cmaera is a bit unreliable due to scale drift...
        camera_position = ground_position + \
            np.array([0, camera_height, 60])  # Slightly behind

        # Fixed orientation (looking down and slightly forward)
        cos_a = np.cos(tilt_angle)
        sin_a = np.sin(tilt_angle)

        R_topdown = np.array([
            [1,     0,      0],
            [0, cos_a, -sin_a],
            [0, sin_a,  cos_a]
        ])

        # Create camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R_topdown
        camera_pose[:3, 3] = camera_position

        self.camera_parameters.extrinsic = camera_pose
        self.ctrl.convert_from_pinhole_camera_parameters(
            self.camera_parameters, allow_arbitrary=True)

    def _draw_camera_object(self, R, t, size=1.0):
        _w, _h, _cx, _cy, _f = self.W, self.H, self.K[0,
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

    def _get_tracking_quality_color(self, tracking_quality):
        quality = np.clip(tracking_quality, 0.0, 1.0)

        # Hue from red (0) to green (60) in OpenCV scale (0–179)
        # green is at 60 (approx 120 deg standard HSV)
        hue = int((60 * quality))
        saturation = 230  # in [0, 255]
        value = 230       # in [0, 255]

        hsv_pixel = np.uint8([[[hue, saturation, value]]])  # shape (1, 1, 3)
        rgb_pixel = cv.cvtColor(hsv_pixel, cv.COLOR_HSV2RGB)[0][0]

        return [float(c) / 255.0 for c in rgb_pixel]

    def _remove_deleted_poses(self, current_keyframe_ids):
        """Remove pose geometries for keyframes that are no longer in the map."""
        poses_to_remove = []

        for keyframe_id in self.poses.keys():
            if keyframe_id not in current_keyframe_ids:
                poses_to_remove.append(keyframe_id)

        for keyframe_id in poses_to_remove:
            debug_log(
                LOG_TAG, f"Removing pose geometry for deleted frame {keyframe_id}")
            pose_geometry = self.poses[keyframe_id][0]

            # Remove from visualizer
            self.vis.remove_geometry(pose_geometry, False)

            # Remove from poses dictionary
            del self.poses[keyframe_id]

        if poses_to_remove:
            debug_log(
                LOG_TAG, f"Removed {len(poses_to_remove)} pose geometries")

    def remove_pose_by_id(self, keyframe_id):
        if keyframe_id in self.poses:
            debug_log(
                LOG_TAG, f"Manually removing pose geometry for frame {keyframe_id}")
            pose_geometry = self.poses[keyframe_id][0]

            # Remove from visualizer
            self.vis.remove_geometry(pose_geometry, False)

    # ======= Utility methods used for output data management =======
    def start_recording_session(self, filename=None):
        if filename:
            self.video_filename = filename
            if not filename.endswith('.mp4'):
                self.video_filename += '.mp4'
        else:
            self.video_filename = os.path.join(
                self.output_path, f"slam_session_{self.timestamp}.mp4")

        self._start_recording()

    def save_session_data(self):
        # Save session metadata
        self.session_data['timestamp'] = self.timestamp
        self.session_data['camera_params'] = {
            'K': self.K.tolist(),
            'width': self.W,
            'height': self.H
        }

        filename = f"slam_session_{self.timestamp}"

        # Save as JSON for metadata and poses
        json_filename = os.path.join(self.output_path, f"{filename}.json")
        with open(json_filename, 'w') as f:
            json.dump(self.session_data, f, indent=2)

        # Save point cloud as PLY file
        ply_filename = os.path.join(self.output_path, f"{filename}.ply")
        if len(self.point_cloud.points) > 0:
            o3d.io.write_point_cloud(ply_filename, self.point_cloud)
            info_log(LOG_TAG, f"Saved point cloud to {ply_filename}")

        info_log(
            LOG_TAG, f"Session saved: {json_filename}, {ply_filename}")

    def get_recording_status(self):
        return {
            'recording': self.recording,
            'filename': self.video_filename,
            'frame_count': self.frame_count
        }

    def _toggle_recording(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()
        return False

    def _save_session(self):
        self.save_session_data()
        return False

    def _start_recording(self):
        self.recording = True
        self.frame_count = 0
        info_log(LOG_TAG, f"Started recording to {self.video_filename}")

    def _stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.recording = False
        info_log(
            LOG_TAG, f"Stopped recording. Saved {self.frame_count} frames to {self.video_filename}")
        self.frame_count = 0

    def capture_frame(self, complement_frame=None):
        if self.frame_count == 0:
            scale = 2 if complement_frame is not None else 1
            W, H = (self.W * scale, self.H)
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv.VideoWriter(
                self.video_filename, fourcc, 30.0, (W, H))

        # Capture screen from Open3D visualizer
        renderer_frame = self.vis.capture_screen_float_buffer(False)
        renderer_frame = np.asarray(renderer_frame)
        renderer_frame = (renderer_frame * 255).astype(np.uint8)
        renderer_frame = cv.cvtColor(renderer_frame, cv.COLOR_RGB2BGR)
        save_img = np.concatenate([complement_frame, renderer_frame], axis=1)

        # Write frame to video
        self.video_writer.write(save_img)
        self.frame_count += 1
