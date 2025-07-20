import cv2 as cv
import numpy as np
from typing import Dict, Callable, Tuple, List

from slam.features.super_point_extractor import SuperPointFrontend
from slam.utils.constants import *
from slam.utils.logger import debug_log

LOG_TAG = 'FeatureExtractor'

DEFAULT_ORB_CONFIG = {
    'n_pts': ORB_NUMBER_OF_POINTS,
    'quality_level': ORB_QUALITY_LEVEL,
    'min_distance': ORB_MIN_DISTANCE
}


class FeatureExtractor:
    def __init__(self, feature_extraction_method: str = ORB_EXTRACTOR_NAME, orb_config: Dict = DEFAULT_ORB_CONFIG):
        self.feature_extraction_method = feature_extraction_method
        self.orb_config = orb_config
        self.extract_handler = None

        match feature_extraction_method:
            case "ORB":
                self.detector = cv.ORB_create()
                self.extract_handler = self._extract_orb
            case "A-KAZE":
                self.detector = cv.AKAZE_create()
                self.extract_handler = self._extract_akaze
            case "DNN":
                self.detector = SuperPointFrontend(
                    weights_path="weights/superpoint_v1.pth",
                    nms_dist=4,
                    conf_thresh=0.00000005,
                    # conf_thresh=0.000000015,
                )
                self.extract_handler = self._extract_dnn
            case _:
                raise ValueError(
                    f"Unsupported feature extraction method: {feature_extraction_method}")

        debug_log(
            LOG_TAG, f"Initialized {feature_extraction_method} feature extractor")

    def extract(self, img: np.ndarray) -> Callable[[np.ndarray], Tuple]:
        return self.extract_handler(img)

    def _extract_dnn(self, img: np.ndarray) -> Tuple[List, List]:
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.
        pts, desc, _ = self.detector.compute(img)
        if pts.shape[1] == 0:
            return [], np.array([])
        keypoints = [
            cv.KeyPoint(
                x=pts[0, i],
                y=pts[1, i],
                size=10,
                angle=-1,
                response=pts[2, i],
                octave=0,
                class_id=-1
            ) for i in range(pts.shape[1])]
        descriptors = desc.T

        return keypoints, descriptors

    def _extract_orb(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = cv.goodFeaturesToTrack(
            img,
            self.orb_config["n_pts"],
            qualityLevel=self.orb_config["quality_level"],
            minDistance=self.orb_config["min_distance"]
        )
        if pts is None:
            return [], None
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=ORB_KEYPOINT_SIZE)
               for f in pts]
        keypoints, descriptors = self.detector.compute(img, kps)
        debug_log(LOG_TAG, f"Extracted {len(keypoints)} ORB features")

        return keypoints, descriptors

    def _extract_akaze(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(
            img.shape) == 3 else img
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        debug_log(LOG_TAG, f"Extracted {len(keypoints)} A-KAZE features")

        return keypoints, descriptors
