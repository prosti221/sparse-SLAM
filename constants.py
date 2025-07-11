# State estimator constants
MIN_DEPTH = 0.1  # Minimum depth for valid points
MAX_DEPTH = 1500.0  # Maximum depth for valid points
MIN_BASELINE_THRESHOLD = 0.1  # Minimum baseline for matching
MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES = 10  # Max frames before new keyframe
NEW_POINTS_THRESHOLD = 0.16  # Threshold for adding new points
MINIMUM_TRANSLATION_THRESHOLD = 1.8  # Minimum translation for new keyframe
MINIMUM_ROTATION_THRESHOLD = 0.1  # Minimum rotation for new keyframe
MINIMUM_PARALLAX_THRESHOLD = 0.45
# Minimum inliers for new keyframe
MINIMUM_NUMBER_OF_INLIERS_FOR_NEW_KEYFRAME = 20
PNP_ITERATIONS_COUNT = 100  # Iterations for PnP RANSAC
PNP_REPROJECTION_ERROR = 8.0  # Reprojection error for PnP RANSAC
PNP_MINIMUM_INLIERS = 6  # Minimum inliers for PnP
MAX_SQUARED_REPROJECTION_ERROR = 4.0
# Window size for local bundle adjustment
LOCAL_BUNDLE_ADJUSTMENT_WINDOW_SIZE = 10
GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL = 20  # Interval for global BA


# Feature extractor constants
ORB_NUMBER_OF_POINTS = 4000
ORB_QUALITY_LEVEL = 0.01  # Quality level for feature detection
ORB_MIN_DISTANCE = 7  # Minimum distance between features
ORB_KEYPOINT_SIZE = 20  # Size of keypoints for feature extraction


# Feature matching constants
KNN_K_VALUE = 2  # Number of nearest neighbors for KNN matching
LOWE_RATIO = 0.7  # Lowe's ratio test threshold for feature matching
LOWE_DISTANCE_THRESHOLD = 32  # Distance threshold for good matches
# RANSAC threshold for essential matrix estimation
MATCHER_RANSAC_MINIMUM_INLIERS = 8  # Minimum inliers for RANSAC
MATCHER_RANSAC_THRESHOLD = 0.005
MATCHER_RANSAC_PROBABILITY = 0.999  # Probability for RANSAC

# Renderer constants


# Bundle adjustment constants
MINIMUM_LOCAL_KEYFRAMES = 2  # Minimum keyframes for local BA
MINIMUM_LOCAL_POINTS = 10  # Minimum points for local BA
MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT = 20
OUTLIER_LOCAL_ERROR_THRESHOLD_FOR_POINTS = 2.5  # Threshold for outlier points
# Threshold for number of ovservations needed to consider a point as outlier
OUTLIER_LOCAL_OBSERVATIONS_THRESHOLD_FOR_POINTS = 2
LEAST_SQUARES_F_SCALE = 1.0  # Scale for least squares optimization

MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT = 50
OUTLIER_GLOBAL_ERROR_THRESHOLD_FOR_POINTS = 2.5
OUTLIER_GLOBAL_OBSERVATIONS_THRESHOLD_FOR_POINTS = 3
