import numpy as np
import cv2 as cv


class Frame:
    def __init__(self, image, K, frame_id=None):
        self.frame_id = frame_id
        self.image = image

        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        self.keypoints = None
        self.descriptors = None

        self.pose = np.eye(4)

        self.matches = None
        self.matched_pts = None      # Points matched in this frame
        self.matched_pts_prev_frame = None  # Corresponding points in previous frame
        self.matched_pts_colors = None  # Colors of matched points in this frame

    def set_features(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors

    def set_match_data(self, matches, matched_pts, matched_pts_prev_frame):
        self.matches = matches
        self.matched_pts = matched_pts
        self.matched_pts_prev_frame = matched_pts_prev_frame

        self.__set_color_values_for_matched_points()

    def keypoints_descriptors(self):
        return self.keypoints, self.descriptors

    def get_gray_image(self):
        return cv.cvtColor(self.image, cv.IMREAD_GRAYSCALE)

    def __set_color_values_for_matched_points(self):
        # Sets the color values for the matched points in the current frame
        if self.matched_pts is not None:
            self.matched_pts_colors = np.array(
                [self.image[int(pt[1]), int(pt[0])][::-1] for pt in self.matched_pts])
        else:
            self.matched_pts_colors = None
