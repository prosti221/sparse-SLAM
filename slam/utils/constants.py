# --- State estimator constants
MINIMUM_NUMBER_OF_INLIERS_FOR_PROJECTION_MATCHING = 10
PNP_ITERATIONS_COUNT = 100  # Iterations for PnP RANSAC
PNP_REPROJECTION_ERROR = 8.0  # Reprojection error for PnP RANSAC
PNP_MINIMUM_INLIERS = 6  # Minimum inliers for PnP

### KEYFRAME SELECTION CONSTANTS ###
MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES = 3
MINIMUM_BASELINE_THRESHOLD = 0.5  # Minimum translation for new keyframe
# Minimum scene coverage for new keyframe
MINIMUM_SCENE_COVERAGE_THRESHOLD = 0.30
MINIMUM_KEYFRAME_PARALLAX_THRESHOLD = 1  # Minimum parallax for new keyframe
MINIMUM_KEYFRAME_QUALITY_THRESHOLD = 0.1  # Minimum quality for new keyframe
# Minimum number of new points for keyframe selection
MINIMUM_NUMBER_OF_NEW_POINTS = 0.4

# Triangulation constants
MIN_DEPTH = 0.0  # Minimum depth for valid points
MAX_DEPTH = 1500.0  # Maximum depth for valid points
MAX_SQUARED_REPROJECTION_ERROR = 2.0
MINIMUM_TRIANGULATION_PARALLAX_THRESHOLD = 0.30

# Window size for local bundle adjustment
LOCAL_MAP_WINDOW_SIZE = 20
GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL = 20  # Interval for global BA

# Bidirectional validation codes
TRIANGULATION_VALIDATION_CODE = {
    'VALID': 0,
    'MIN_DEPTH_VIOLATION': 1,
    'MAX_DEPTH_VIOLATION': 2,
    'MIN_BASELINE_VIOLATION': 3,
    'MAX_REPROJECTION_ERROR_VIOLATION': 4,
    'MIN_PARALLAX_VIOLATION': 5,
    'WELL_CONDITIONED_VIOLATION': 6
}
TRIANGULATION_VALIDATION_CODE.update(
    dict([reversed(i) for i in TRIANGULATION_VALIDATION_CODE.items()]))


# --- Feature extractor constants
ORB_EXTRACTOR_NAME = 'ORB'
DNN_EXTRACTOR_NAME = 'DNN'
AKAZE_EXTRACTOR_NAME = 'A-KAZE'
BINARY_DESCRIPTION_METHODS = [ORB_EXTRACTOR_NAME, AKAZE_EXTRACTOR_NAME]

ORB_NUMBER_OF_POINTS = 5000
ORB_QUALITY_LEVEL = 0.01  # Quality level for feature detection
ORB_MIN_DISTANCE = 7  # Minimum distance between features
ORB_KEYPOINT_SIZE = 20  # Size of keypoints for feature extraction


# --- Feature matching constants
KNN_K_VALUE = 2  # Number of nearest neighbors for KNN matching
LOWE_RATIO = 0.75  # Lowe's ratio test threshold for feature matching
# RANSAC threshold for essential matrix estimation
MATCHER_RANSAC_MINIMUM_INLIERS = 8  # Minimum inliers for RANSAC
MATCHER_RANSAC_THRESHOLD = 0.02
MATCHER_RANSAC_PROBABILITY = 0.999  # Probability for RANSAC

# --- Bundle adjustment constants
MINIMUM_LOCAL_KEYFRAMES = 2  # Minimum keyframes for local BA
MINIMUM_LOCAL_POINTS = 10  # Minimum points for local BA
MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT = 20
OUTLIER_LOCAL_ERROR_THRESHOLD_FOR_POINTS = 6  # Threshold for outlier points
# Threshold for number of ovservations needed to consider a point as outlier
OUTLIER_LOCAL_OBSERVATIONS_THRESHOLD_FOR_POINTS = 2
LEAST_SQUARES_F_SCALE = 0.2  # Scale for least squares optimization

MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT = 50
OUTLIER_GLOBAL_ERROR_THRESHOLD_FOR_POINTS = 6
OUTLIER_GLOBAL_OBSERVATIONS_THRESHOLD_FOR_POINTS = 3

# --- Renderer constants
TRACKING_QUALITY_GRADIENT = {
    'min_hue': 0.0,    # Red (poor tracking)
    'max_hue': 0.3,    # Green (excellent tracking)
    'saturation': 0.9,
    'value': 0.9
}


# --- Map constants
TRACKING_QUALITY_WINDOW_SIZE = 5
TRACKING_QUALITY_THRESHOLD = 0.2
