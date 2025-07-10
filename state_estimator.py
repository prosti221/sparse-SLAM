"""
The state estimator will only take in cur features, and use these to compute all of our states.
"""
import numpy as np
import cv2 as cv
from utils import *
from matcher import *
from point import Point


class StateEstimator:
    def __init__(self):
        self.cur_frame = None
        self.prev_frame = None

    def update(self, new_frame):
        if not self.cur_frame:
            self.cur_frame = new_frame
            return

        self.prev_frame = self.cur_frame
        self.cur_frame = new_frame

        # Find feature correspondence
        matches, F = match_features(
            self.prev_frame, self.cur_frame)

        # Store matches in current frame
        matched_pts = np.float64(
            [self.cur_frame.keypoints[m.trainIdx].pt for m in matches])
        prev_frame_matched_pts = np.float64(
            [self.prev_frame.keypoints[m.queryIdx].pt for m in matches])

        self.cur_frame.set_match_data(
            matches, matched_pts, prev_frame_matched_pts)

        Rt = extractRt(F)
        self.cur_frame.pose = Rt @ self.prev_frame.pose

    def get_camera_pose(self):
        return self.cur_frame.pose if self.cur_frame else None

    def compute(self):
        ret = np.zeros((self.cur_frame.matched_pts.shape[0], 4))

        pose1 = np.linalg.inv(self.cur_frame.pose)
        pose2 = np.linalg.inv(self.prev_frame.pose)

        pts1 = normalize(self.cur_frame.matched_pts, self.cur_frame.Kinv)
        pts2 = normalize(self.cur_frame.matched_pts_prev_frame,
                         self.prev_frame.Kinv)

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
        if not (self.cur_frame and self.prev_frame):
            print("Not enough frames to triangulate points.")
            return []

        # Triangulate points
        points_4d = self.compute()
        points_4d = points_4d / points_4d[:, 3:]

        points = []
        for i, m in enumerate(self.cur_frame.matches):
            point_4d = points_4d[i]

            pl1 = np.dot(self.cur_frame.pose, points_4d[i])
            pl2 = np.dot(self.prev_frame.pose, points_4d[i])
            if pl1[2] < 0 or pl2[2] < 0 or np.abs(point_4d[3]) < 0.005:
                continue

            # Reprojection error
            pp1 = np.dot(self.cur_frame.K, pl1[:3])
            pp2 = np.dot(self.cur_frame.K, pl2[:3])

            # check reprojection error
            pp1 = (pp1[0:2] / pp1[2]) - \
                self.cur_frame.keypoints_descriptors()[0][m.trainIdx].pt
            pp2 = (pp2[0:2] / pp2[2]) - \
                self.prev_frame.keypoints_descriptors()[0][m.queryIdx].pt

            pp1 = np.sum(pp1**2)
            pp2 = np.sum(pp2**2)

            if pp1 > 25 or pp2 > 25:
                continue

            points.append(
                Point(np.array(point_4d)[:3], self.cur_frame.matched_pts_colors[i]))

        print(f"Triangulated {len(points)} points.")

        return points

    def visualize_matches(self):
        # Draws a line between the matching points on the same image
        if self.cur_frame:
            img_cpy = self.cur_frame.get_gray_image().copy()
            for m in self.cur_frame.matches:
                pt1 = tuple(
                    map(int, self.prev_frame.keypoints_descriptors()[0][m.queryIdx].pt))
                pt2 = tuple(
                    map(int, self.cur_frame.keypoints_descriptors()[0][m.trainIdx].pt))
                cv.line(img_cpy, pt1, pt2, (0, 255, 0), 1)
            cv.imshow('matches', img_cpy)
        else:
            print("No matches to visualize... skipping.")
