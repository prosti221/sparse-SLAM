"""
The state estimator will only take in cur features, and use these to compute all of our states.
"""
import numpy as np
import cv2 as cv
from utils import *
from matcher import *
import warnings

class Point:
    def __init__(self, pt_2d=None, color=None, pt_3d=None):
        """
        A point in 3D space. It will have its own 3D coordinates, and a color.
        """
        self.pt_2d = pt_2d
        self.pt_3d = pt_3d 
        self.color = color

class StateEstimator:
    def __init__(self, K):
        self.frame = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        self.cur_pose = None
        self.prev_pose = None

        self.prev_features = None
        self.cur_features = None

        self.filtered_matches = None
        self.prev_filtered_pts = None
        self.cur_filtered_pts = None

    def update(self, features, frame):
        cur_kp, _ = features
        if not self.cur_features:
            self.cur_features = features
            return

        self.frame = frame
        self.prev_features = self.cur_features
        self.cur_features = features

        # Find feature correspondence
        self.filtered_matches, E = match_features(self.prev_features, self.cur_features, self.K)
                
        self.prev_filtered_pts = normalize(np.float64([self.prev_features[0][m.queryIdx].pt for m in self.filtered_matches]), self.Kinv)
        self.cur_filtered_pts = normalize(np.float64([self.cur_features[0][m.trainIdx].pt for m in self.filtered_matches]), self.Kinv)

        Rt = extractRt(E)

        self.prev_pose = self.cur_pose
        self.cur_pose = Rt
    
    def get_camera_pose(self):
        return self.cur_pose
    
    def compute(self, pose1, pose2, pts1, pts2):
        ret = np.zeros((pts1.shape[0], 4))
        pose1 = np.linalg.inv(pose1)
        pose2 = np.linalg.inv(pose2)

        pts1 = add_ones(np.squeeze(pts1))
        pts2 = add_ones(np.squeeze(pts2))

        for i, p in enumerate(zip(pts1, pts2)):
            A = np.zeros((4, 4))
            A[0] = p[0][0] * pose1[2] - pose1[0]
            A[1] = p[0][1] * pose1[2] - pose1[1]
            A[2] = p[1][0] * pose2[2] - pose2[0]
            A[3] = p[1][1] * pose2[2] - pose2[1]

            _, _, vt = np.linalg.svd(A)
            ret[i] = vt[3]

        return ret
    
    def triangulate(self):
        # Update current pose
        self.cur_pose = self.cur_pose @ self.prev_pose

        # Triangulate points
        points_4d = self.compute(self.cur_pose, self.prev_pose, self.cur_filtered_pts, self.prev_filtered_pts)
        points_4d = points_4d / points_4d[:, 3:]

        points = []
        for i, m in enumerate(self.filtered_matches):
            point_2d = self.cur_features[0][m.trainIdx].pt
            point_4d = points_4d[i]
            if np.abs(point_4d[3]) < 0.005 or point_4d[2] < 0:
                continue
            color = self.frame[int(point_2d[1]), int(point_2d[0])][::-1]
            points.append(Point(point_2d, color, np.array(point_4d)[:3]))
            
        return points

    def visualize_matches(self, cur_img):
        # Draws a line between the matching points on the same image
        if self.filtered_matches:
            img_cpy = cur_img.copy()
            for m in self.filtered_matches:
                pt1 = tuple(map(int, self.prev_features[0][m.queryIdx].pt))
                pt2 = tuple(map(int, self.cur_features[0][m.trainIdx].pt))
                cv.line(img_cpy, pt1, pt2, (0, 255, 0), 1)
            cv.imshow('matches', img_cpy)
        else:
            warnings.warn("No matches to visualize... skipping.")