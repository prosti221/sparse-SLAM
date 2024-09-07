import numpy as np
import cv2 as cv

def pt_obj_to_array(pts):
    points_3d = np.array([pt.pt_3d for pt in pts])
    color = np.array([pt.color for pt in pts])

    color = color / 255.0
    return points_3d, color

def get_median_pixel_color(frame, pt, k_size=3):
    """
    Given a pixel in the frame, return the median pixel value in a k_size x k_size window.
    """
    x, y = pt
    x, y = int(x), int(y)
    if x < k_size or y < k_size or x >= frame.shape[1] - k_size or y >= frame.shape[0] - k_size:
        return frame[y, x]
    
    window = frame[y-k_size:y+k_size, x-k_size:x+k_size]
    return np.median(window, axis=(0, 1))

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize(pts, Kinv):
    return np.dot(Kinv, add_ones(pts).T).T[:, :2]

def denormalize(points, K):
		ret = np.dot(K, np.array([points[0], points[1], 1.0]))
		return int(round(ret[0])), int(round(ret[1]))

def extractRt(E):
	W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
	U, d, Vt = np.linalg.svd(E)
	if np.linalg.det(U) < 0:
		U *= -1.0
	if np.linalg.det(Vt) < 0:
		Vt *= -1.0
	
	R = np.dot(np.dot(U, W), Vt)
	if np.sum(R.diagonal()) < 0:
		R = np.dot(np.dot(U, W.T), Vt)

	t = U[:, 2]
	ret = np.eye(4)
	ret[:3, :3] = R
	ret[:3, 3] = t

	return ret

def get_video_cap(path):
    #FRAME_RATE = 15
    cap = cv.VideoCapture(path)
    #cap.set(cv.CAP_PROP_FPS, FRAME_RATE)

    return cap

def construct_K(focal_x, focal_y, c_x, c_y):

    return np.array([[focal_x, 0, c_x],
                     [0, focal_y, c_y],
                     [0, 0, 1]])

def draw_keypoints(frame, keypoints):
    img_cpy = frame.copy()
    keypoint_frame = cv.drawKeypoints(
        frame, keypoints, img_cpy, color=(0, 255, 0), flags=0
        )

    return img_cpy

def draw_matches(prev_img, prev_keypoints, cur_img, cur_keypoints, matches):
    prev_img = prev_img.copy()
    cur_img = cur_img.copy()

    match_img = cv.drawMatchesKnn(
        prev_img,prev_keypoints, 
        cur_img,cur_keypoints,
        [matches],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return match_img