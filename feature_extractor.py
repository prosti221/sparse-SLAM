import cv2 as cv
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # TODO: Add more feature extractors to choose from
        self.detector = cv.ORB_create()
    
    def extract(self, frame, n_pts=4000, quality_level=0.01, min_distance=7):
        pts = cv.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), n_pts, qualityLevel=quality_level, minDistance=min_distance)

        # extraction
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
        features = self.detector.compute(frame, kps)

        return features # keypoints, descriptors