import cv2 as cv
import numpy as np
from constants import *


class FeatureExtractor:
    def __init__(self):
        # TODO: Add more feature extractors to choose from
        self.detector = cv.ORB_create()

        # TODO: Make these parameters configurable
        self.n_pts = ORB_NUMBER_OF_POINTS
        self.quality_level = ORB_QUALITY_LEVEL
        self.min_distance = ORB_MIN_DISTANCE

    def extract(self, frame):
        pts = cv.goodFeaturesToTrack(np.mean(frame, axis=2).astype(
            np.uint8), self.n_pts, qualityLevel=self.quality_level, minDistance=self.min_distance)

        # extraction
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=ORB_KEYPOINT_SIZE)
               for f in pts]
        features = self.detector.compute(frame, kps)

        return features  # keypoints, descriptors
