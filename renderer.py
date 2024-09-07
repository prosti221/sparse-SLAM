import open3d as o3d
import numpy as np
from utils import pt_obj_to_array

class Renderer:
    def __init__(self, K, width=1920, height=1080):
        self.vis = o3d.visualization.Visualizer()

        self.width = width
        self.height = height 

        self.K = K

        self.point_cloud = o3d.geometry.PointCloud()

        self.ctrl = None
        self.cloud_initialized = False
        self.camera_initialized = False

        self.pinhole = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self.camera_parameters = o3d.camera.PinholeCameraParameters()
        self.camera_parameters.intrinsic = self.pinhole

    def update_points(self, pts):
        if len(pts) == 0: return
        pts, colors = pt_obj_to_array(pts)

        if not self.cloud_initialized:
            self.point_cloud.points = o3d.utility.Vector3dVector(pts)
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(self.point_cloud)
            self.cloud_initialized = True
        else:
            self.point_cloud.points.extend(o3d.utility.Vector3dVector(pts))
            self.point_cloud.colors.extend(o3d.utility.Vector3dVector(colors))

        # Update view
        self.vis.get_render_option().point_size = 1.5
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def start(self):
        self.vis.create_window(window_name="SLAM", width=self.width, height=self.height)
        self.vis.get_render_option().background_color = [0.0, 0.0, 0.0]
        self.ctrl = self.vis.get_view_control()
        self.ctrl.convert_from_pinhole_camera_parameters(self.camera_parameters, allow_arbitrary=True)
          
    def stop(self):
        self.vis.destroy_window()
    
    def update_camera(self, Rt):
        # Update cumulative position
        R, t = Rt[:3, :3], Rt[:3, 3]
        points, lines = self.draw_camera_object(R, t)

        new_cam = o3d.geometry.LineSet()
        new_cam.points = points
        new_cam.lines = lines
        # Set color to red
        colors = np.zeros((len(lines), 3))
        colors[:, 1] = 1
        new_cam.colors = o3d.utility.Vector3dVector(colors)

        if not self.camera_initialized:
            Rt[:3, 3] += 20 * R[:, 2]
            self.camera_parameters.extrinsic = Rt 
            self.ctrl.convert_from_pinhole_camera_parameters(self.camera_parameters, allow_arbitrary=True)
            self.ctrl.set_constant_z_near(10)
            #self.ctrl.set_constant_z_far(1000)
            self.camera_initialized = True
            
        self.vis.add_geometry(new_cam, False)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def draw_camera_object(self, R, t, size=1.0):
        _w, _h, _cx, _cy, _f = self.width, self.height , self.K[0, 2], self.K[1, 2], self.K[0, 0]
        f = 1
        w = _w/_f
        h = _h/_f
        cx = _cx/_f
        cy = _cy/_f

        offset_cx = cx - w/2.0
        offset_cy = cy - h/2.0

        points = [[0, 0, 0],
          [offset_cx,offset_cy, f],
          [-0.5 * w, -0.5 * h, f],
          [ 0.5 * w, -0.5 * h, f],
          [ 0.5 * w, 0.5 * h, f],
          [-0.5 * w, 0.5 * h, f],
          [-0.5 * w, -0.5 * h, f]]

        lines = [[0,1],[2,3],[3,4],[4,5],[5,6],[0,2],[0,3],[0,4],[0,5],[2,4],[3,5]]

        points = np.array(points) * size

        points = (R @ points.T).T + t

        points = o3d.utility.Vector3dVector(points)
        lines = o3d.utility.Vector2iVector(lines)

        return points, lines