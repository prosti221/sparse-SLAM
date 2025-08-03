import cv2 as cv
import numpy as np
from typing import Dict, Callable, Tuple, List

from slam.features.super_point import SuperPointFrontend
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

DEFAULT_AKAZE_CONFIG = {
    'pyramid_levels': 6,
    'scale_factor': 1.4,
    'multiscale_enabled': True,
    'descriptor_type': cv.AKAZE_DESCRIPTOR_MLDB,
    'descriptor_size': 0,
    'descriptor_channels': 3,
    'threshold': 0.001,
    'n_octaves': 4,
    'n_octave_layers': 4
}

DEFAULT_DNN_CONFIG = {
    'pyramid_levels': 4,
    'scale_factor': 1.3,
    'multiscale_enabled': False,
    'scales': [1.0, 0.8, 0.6, 1.2],
    'weights_path': "weights/superpoint_v1.pth",
    'nms_dist': 4,
    'conf_thresh': 0.000000005
    # 'conf_thresh': 0.0005
}


class FeatureExtractor:
    def __init__(self, feature_extraction_method: str = ORB_EXTRACTOR_NAME,
                 orb_config: Dict = None,
                 akaze_config: Dict = None,
                 dnn_config: Dict = None):
        self.feature_extraction_method = feature_extraction_method

        # Set default configs if not provided
        self.orb_config = {**DEFAULT_ORB_CONFIG, **(orb_config or {})}
        self.akaze_config = {**DEFAULT_AKAZE_CONFIG, **(akaze_config or {})}
        self.dnn_config = {**DEFAULT_DNN_CONFIG, **(dnn_config or {})}

        self.extract_handler = None

        if feature_extraction_method == "ORB":
            # Configure ORB with multiscale support
            self.detector = cv.ORB_create(
                nfeatures=self.orb_config.get('n_pts', 1000),
                scaleFactor=self.orb_config.get('scale_factor', 1.2),
                nlevels=self.orb_config.get('pyramid_levels', 8),
                edgeThreshold=self.orb_config.get('edge_threshold', 31),
                firstLevel=self.orb_config.get('first_level', 0),
                WTA_K=self.orb_config.get('wta_k', 2),
                patchSize=self.orb_config.get('patch_size', 31)
            )

            # Choose extraction method based on multiscale setting
            if self.orb_config.get('multiscale_enabled', True):
                self.extract_handler = self._extract_orb_multiscale
            else:
                self.extract_handler = self._extract_orb_single_scale
        elif feature_extraction_method == "AKAZE":
            # Configure AKAZE with multiscale support
            self.detector = cv.AKAZE_create(
                descriptor_type=self.akaze_config.get(
                    'descriptor_type', cv.AKAZE_DESCRIPTOR_MLDB),
                descriptor_size=self.akaze_config.get(
                    'descriptor_size', 0),
                descriptor_channels=self.akaze_config.get(
                    'descriptor_channels', 3),
                threshold=self.akaze_config.get('threshold', 0.001),
                nOctaves=self.akaze_config.get('n_octaves', 4),
                nOctaveLayers=self.akaze_config.get('n_octave_layers', 4)
            )

            if self.akaze_config.get('multiscale_enabled', True):
                self.extract_handler = self._extract_akaze_multiscale
            else:
                self.extract_handler = self._extract_akaze_single_scale
        elif feature_extraction_method == "DNN":
            self.detector = SuperPointFrontend(
                weights_path=self.dnn_config.get(
                    'weights_path', "weights/superpoint_v1.pth"),
                nms_dist=self.dnn_config.get('nms_dist', 4),
                conf_thresh=self.dnn_config.get('conf_thresh', 0.000000005)
            )

            if self.dnn_config.get('multiscale_enabled', True):
                self.extract_handler = self._extract_dnn_multiscale
            else:
                self.extract_handler = self._extract_dnn_single_scale
        else:
            raise ValueError(
                f"Unsupported feature extraction method: {feature_extraction_method}")

        current_config = getattr(
            self, f"{feature_extraction_method.lower().replace('-', '_')}_config")
        multiscale_info = ""
        if current_config.get('multiscale_enabled', True):
            levels = current_config.get('pyramid_levels', 4)
            multiscale_info = f" with multiscale ({levels} levels)"

        debug_log(
            LOG_TAG, f"Initialized {feature_extraction_method} feature extractor{multiscale_info}")

    def extract(self, img: np.ndarray) -> Callable[[np.ndarray], Tuple]:
        return self.extract_handler(img)

    def _create_image_pyramid(self, img: np.ndarray, n_levels: int, scale_factor: float) -> List[Tuple[np.ndarray, float]]:
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

    def _create_custom_scales(self, img: np.ndarray, scales: List[float]) -> List[Tuple[np.ndarray, float]]:
        scaled_images = []

        for scale in scales:
            if scale == 1.0:
                scaled_images.append((img, scale))
            else:
                new_width = max(1, int(img.shape[1] * scale))
                new_height = max(1, int(img.shape[0] * scale))

                if new_width < 20 or new_height < 20:
                    continue

                scaled_img = cv.resize(
                    img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
                scaled_images.append((scaled_img, scale))

        return scaled_images

    # ===== ORB METHODS =====

    def _extract_orb_multiscale(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        pyramid = self._create_image_pyramid(
            gray,
            self.orb_config.get('pyramid_levels', 8),
            self.orb_config.get('scale_factor', 1.2)
        )

        all_keypoints = []
        all_descriptors = []
        total_points_per_level = self.orb_config['n_pts'] // len(pyramid)

        debug_log(
            LOG_TAG, f"Processing {len(pyramid)} pyramid levels for multiscale ORB")

        for level, (scaled_img, scale) in enumerate(pyramid):
            min_dist = max(1, self.orb_config['min_distance'] / scale)

            # Extract corner points at this scale
            pts = cv.goodFeaturesToTrack(
                scaled_img,
                maxCorners=total_points_per_level,
                qualityLevel=self.orb_config['quality_level'],
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
                keypoints_computed, descriptors = self.detector.compute(
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

    def _extract_orb_single_scale(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        pts = cv.goodFeaturesToTrack(
            gray,
            self.orb_config["n_pts"],
            qualityLevel=self.orb_config["quality_level"],
            minDistance=self.orb_config["min_distance"]
        )

        if pts is None:
            return [], None

        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=ORB_KEYPOINT_SIZE)
               for f in pts]
        keypoints, descriptors = self.detector.compute(gray, kps)
        debug_log(
            LOG_TAG, f"Extracted {len(keypoints)} single-scale ORB features")

        return keypoints, descriptors

    # ===== A-KAZE METHODS =====

    def _extract_akaze_multiscale(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        pyramid = self._create_image_pyramid(
            gray,
            self.akaze_config.get('pyramid_levels', 6),
            self.akaze_config.get('scale_factor', 1.4)
        )

        all_keypoints = []
        all_descriptors = []

        debug_log(
            LOG_TAG, f"Processing {len(pyramid)} pyramid levels for multiscale A-KAZE")

        for level, (scaled_img, scale) in enumerate(pyramid):
            try:
                keypoints, descriptors = self.detector.detectAndCompute(
                    scaled_img, None)

                if keypoints and descriptors is not None:
                    # Scale keypoints back to original image coordinates
                    scaled_keypoints = []
                    for kp in keypoints:
                        scaled_kp = cv.KeyPoint(
                            x=kp.pt[0] * scale,
                            y=kp.pt[1] * scale,
                            size=kp.size * scale,
                            angle=kp.angle,
                            response=kp.response,
                            octave=level,  # Store pyramid level
                            class_id=kp.class_id
                        )

                        # Ensure keypoints are within image bounds
                        if 0 <= scaled_kp.pt[0] < gray.shape[1] and 0 <= scaled_kp.pt[1] < gray.shape[0]:
                            scaled_keypoints.append(scaled_kp)

                    if scaled_keypoints:
                        all_keypoints.extend(scaled_keypoints)
                        all_descriptors.append(
                            descriptors[:len(scaled_keypoints)])

            except cv.error as e:
                debug_log(
                    LOG_TAG, f"Error computing A-KAZE descriptors at level {level}: {e}")
                continue

        # Combine all descriptors
        if all_descriptors:
            final_descriptors = np.vstack(all_descriptors)
            debug_log(
                LOG_TAG, f"Extracted {len(all_keypoints)} multiscale A-KAZE features across {len(pyramid)} levels")
        else:
            final_descriptors = None
            debug_log(LOG_TAG, "No A-KAZE features extracted")

        return all_keypoints, final_descriptors

    def _extract_akaze_single_scale(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(
            img.shape) == 3 else img
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        debug_log(
            LOG_TAG, f"Extracted {len(keypoints)} single-scale A-KAZE features")

        return keypoints, descriptors

    # ===== DNN (SuperPoint) METHODS =====

    def _extract_dnn_multiscale(self, img: np.ndarray) -> Tuple[List, List]:
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Use custom scales for DNN
        scales = self.dnn_config.get('scales', [1.0, 0.8, 0.6, 1.2])
        scaled_images = self._create_custom_scales(img, scales)

        all_keypoints = []
        all_descriptors = []

        debug_log(
            LOG_TAG, f"Processing {len(scaled_images)} scales for multiscale SuperPoint: {scales}")

        for scale_idx, (scaled_img, scale) in enumerate(scaled_images):
            try:
                scaled_img_float = scaled_img.astype(np.float32) / 255.
                pts, desc, _ = self.detector.compute(scaled_img_float)

                if pts.shape[1] > 0:
                    # Scale keypoints back to original coordinates
                    keypoints = []
                    for i in range(pts.shape[1]):
                        kp = cv.KeyPoint(
                            # Scale back to original coordinates
                            x=pts[0, i] / scale,
                            y=pts[1, i] / scale,
                            size=10 * scale,
                            angle=-1,
                            response=pts[2, i],
                            octave=scale_idx,  # Use scale index as octave
                            class_id=-1
                        )

                        # Ensure keypoints are within original image bounds
                        if 0 <= kp.pt[0] < img.shape[1] and 0 <= kp.pt[1] < img.shape[0]:
                            keypoints.append(kp)

                    if keypoints:
                        all_keypoints.extend(keypoints)
                        # Only include descriptors for valid keypoints
                        valid_desc = desc[:, :len(keypoints)].T
                        all_descriptors.append(valid_desc)

            except Exception as e:
                debug_log(
                    LOG_TAG, f"Error computing SuperPoint descriptors at scale {scale}: {e}")
                continue

        # Combine all descriptors
        if all_descriptors:
            final_descriptors = np.vstack(all_descriptors)
            debug_log(
                LOG_TAG, f"Extracted {len(all_keypoints)} multiscale SuperPoint features across {len(scaled_images)} scales")
        else:
            final_descriptors = np.array([])
            debug_log(LOG_TAG, "No SuperPoint features extracted")

        return all_keypoints, final_descriptors

    def _extract_dnn_single_scale(self, img: np.ndarray) -> Tuple[List, List]:
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

        debug_log(
            LOG_TAG, f"Extracted {len(keypoints)} single-scale SuperPoint features")
        return keypoints, descriptors

    def set_multiscale_enabled(self, enabled: bool):
        method = self.feature_extraction_method.lower().replace('-', '_')
        config = getattr(self, f"{method}_config")
        config['multiscale_enabled'] = enabled
        # Update extraction handler
        if self.feature_extraction_method == "ORB":
            self.extract_handler = self._extract_orb_multiscale if enabled else self._extract_orb_single_scale
        elif self.feature_extraction_method == "AKAZE":
            self.extract_handler = self._extract_akaze_multiscale if enabled else self._extract_akaze_single_scale
        elif self.feature_extraction_method == "DNN":
            self.extract_handler = self._extract_dnn_multiscale if enabled else self._extract_dnn_single_scale

        debug_log(
            LOG_TAG, f"Multiscale extraction {'enabled' if enabled else 'disabled'} for {self.feature_extraction_method}")
