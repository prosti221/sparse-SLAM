from functools import cached_property
import numpy as np


class Observation():
    def __init__(self, frame, point, kp_idx):
        self.frame = frame
        self.point = point
        self.kp_idx = kp_idx

    @property
    def point_id(self):
        return self.point.point_id

    @property
    def frame_id(self):
        return self.frame.frame_id

    @cached_property
    def pt_2d(self):
        return np.array(self.frame.keypoints[self.kp_idx].pt)

    @cached_property
    def keypoint(self):
        return self.frame.keypoints[self.kp_idx]

    @cached_property
    def pt_2d_norm(self):
        return np.array(self.frame.kp_pts_norm[self.kp_idx])

    @property
    def pt_3d(self):
        return self.point.pt_3d
