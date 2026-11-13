import cv2 as cv
import numpy as np
from typing import Callable, Dict, List, Tuple

from slam.utils.constants import *
from slam.utils.logger import debug_log

LOG_TAG = 'FeatureExtractor'

DEFAULT_ORB_CONFIG = {
    'n_pts': ORB_NUMBER_OF_POINTS,
    'quality_level': ORB_QUALITY_LEVEL,
    'min_distance': ORB_MIN_DISTANCE,
    'pyramid_levels': 8,
    'scale_factor': 1.3,
    'edge_threshold': 22,
    'multiscale_enabled': True,
    'first_level': 0,
    'wta_k': 2,
    'patch_size': 21
}


def _create_image_pyramid(img: np.ndarray, n_levels: int, scale_factor: float) -> List[Tuple[np.ndarray, float]]:
    pyramid = [(img, 1.0)]

    for level in range(1, n_levels):
        scale = scale_factor ** level
        new_width = max(1, int(img.shape[1] / scale))
        new_height = max(1, int(img.shape[0] / scale))

        # Ensure minimum size
        if new_width < 20 or new_height < 20:
            break

        scaled_img = cv.resize(
            img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
        pyramid.append((scaled_img, scale))

    return pyramid


# ===== the extractor registry =====
# One builder per extractor name. A builder owns its DEFAULT config,
# merges the caller's overrides, and returns the merged config, the
# detector, and the (multiscale, single-scale) extraction handlers as
# closures. FeatureExtractor dispatches through this dict only, so a
# new extractor is ONE builder plus ONE
# constants.EXTRACTOR_DESCRIPTOR_KINDS entry (the matcher's norm
# seam) - no class or call-site edits.
#
# AKAZE and SuperPoint ("DNN") were removed: neither ever won a
# measured adoption gate, and every tier constructs ORB. The
# IMPROVEMENT.md record holds the history.

def _build_orb(config: Dict):
    merged = {**DEFAULT_ORB_CONFIG, **(config or {})}

    # Configure ORB with multiscale support
    detector = cv.ORB_create(
        nfeatures=merged.get('n_pts', 1000),
        scaleFactor=merged.get('scale_factor', 1.2),
        nlevels=merged.get('pyramid_levels', 8),
        edgeThreshold=merged.get('edge_threshold', 31),
        firstLevel=merged.get('first_level', 0),
        WTA_K=merged.get('wta_k', 2),
        patchSize=merged.get('patch_size', 31)
    )

    def multiscale(img: np.ndarray) -> Tuple[List, np.ndarray]:
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        pyramid = _create_image_pyramid(
            gray,
            merged.get('pyramid_levels', 8),
            merged.get('scale_factor', 1.2)
        )

        all_keypoints = []
        all_descriptors = []
        total_points_per_level = merged['n_pts'] // len(pyramid)

        debug_log(
            LOG_TAG, f"Processing {len(pyramid)} pyramid levels for multiscale ORB")

        for level, (scaled_img, scale) in enumerate(pyramid):
            min_dist = max(1, merged['min_distance'] / scale)

            # Extract corner points at this scale
            pts = cv.goodFeaturesToTrack(
                scaled_img,
                maxCorners=total_points_per_level,
                qualityLevel=merged['quality_level'],
                minDistance=min_dist,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04
            )

            if pts is None or len(pts) == 0:
                continue

            # Create keypoints scaled back to original image coordinates
            level_keypoints = []
            for pt in pts:
                x_scaled = pt[0][0] * scale
                y_scaled = pt[0][1] * scale

                # Ensure keypoints are within image bounds
                if 0 <= x_scaled < gray.shape[1] and 0 <= y_scaled < gray.shape[0]:
                    kp = cv.KeyPoint(
                        x=x_scaled,
                        y=y_scaled,
                        size=ORB_KEYPOINT_SIZE * scale,
                        angle=-1,
                        response=1.0,
                        octave=level,
                        class_id=-1
                    )
                    level_keypoints.append(kp)

            if not level_keypoints:
                continue

            try:
                keypoints_computed, descriptors = detector.compute(
                    gray, level_keypoints)

                if descriptors is not None and len(keypoints_computed) > 0:
                    all_keypoints.extend(keypoints_computed)
                    all_descriptors.append(descriptors)

            except cv.error as e:
                debug_log(
                    LOG_TAG, f"Error computing ORB descriptors at level {level}: {e}")
                continue

        # Combine all descriptors
        if all_descriptors:
            final_descriptors = np.vstack(all_descriptors)
            debug_log(
                LOG_TAG, f"Extracted {len(all_keypoints)} multiscale ORB features across {len(pyramid)} levels")
        else:
            final_descriptors = None
            debug_log(LOG_TAG, "No ORB features extracted")

        return all_keypoints, final_descriptors

    def single(img: np.ndarray) -> Tuple[List, np.ndarray]:
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        pts = cv.goodFeaturesToTrack(
            gray,
            merged["n_pts"],
            qualityLevel=merged["quality_level"],
            minDistance=merged["min_distance"]
        )

        if pts is None:
            return [], None

        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=ORB_KEYPOINT_SIZE)
               for f in pts]
        keypoints, descriptors = detector.compute(gray, kps)
        debug_log(
            LOG_TAG, f"Extracted {len(keypoints)} single-scale ORB features")

        return keypoints, descriptors

    return merged, detector, multiscale, single


def _build_aliked(config: Dict):
    """The learned frontend. The models live in
    aliked_backend (lazy torch/lightglue imports - the geocalib
    pattern); the builder only wires the handlers. ALIKED handles
    scale internally (the deformable detector is natively
    multi-scale), so the multiscale/single handlers are the SAME
    closure and enable_multiscale_features is a documented no-op -
    the config still records it for the multiscale_info log line.
    Detector = None: the backend owns the model singletons."""
    merged = {**(config or {}),
              'multiscale_enabled': True,  # internal to ALIKED
              'pyramid_levels': 'internal'}

    def extract_handler(img: np.ndarray) -> Tuple[List, np.ndarray]:
        from slam.features import aliked_backend
        kps, desc = aliked_backend.extract_aliked(img)
        debug_log(LOG_TAG, f"Extracted {len(kps)} ALIKED features")
        return kps, desc

    return merged, None, extract_handler, extract_handler


_EXTRACTOR_BUILDERS = {
    'ORB': _build_orb,
    'ALIKED': _build_aliked,
}


class FeatureExtractor:
    def __init__(self, feature_extraction_method: str = ORB_EXTRACTOR_NAME,
                 config: Dict = None):
        self.feature_extraction_method = feature_extraction_method

        builder = _EXTRACTOR_BUILDERS.get(feature_extraction_method)
        if builder is None:
            raise ValueError(
                f"Unsupported feature extraction method: {feature_extraction_method}")

        self.config, self.detector, self._extract_multiscale, self._extract_single = builder(config)

        # Choose extraction method based on multiscale setting
        self.extract_handler = (
            self._extract_multiscale
            if self.config.get('multiscale_enabled', True)
            else self._extract_single)

        multiscale_info = ""
        if self.config.get('multiscale_enabled', True):
            levels = self.config.get('pyramid_levels', 4)
            multiscale_info = f" with multiscale ({levels} levels)"

        debug_log(
            LOG_TAG, f"Initialized {feature_extraction_method} feature extractor{multiscale_info}")

    def extract(self, img: np.ndarray) -> Callable[[np.ndarray], Tuple]:
        return self.extract_handler(img)

    def set_multiscale_enabled(self, enabled: bool):
        self.config['multiscale_enabled'] = enabled
        # Update extraction handler
        self.extract_handler = self._extract_multiscale if enabled else self._extract_single

        debug_log(
            LOG_TAG, f"Multiscale extraction {'enabled' if enabled else 'disabled'} for {self.feature_extraction_method}")
