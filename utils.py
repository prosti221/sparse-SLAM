import numpy as np
import cv2 as cv

def get_video_cap(path):
    FRAME_RATE = 30
    cap = cv.VideoCapture(path)
    cap.set(cv.CAP_PROP_FPS, FRAME_RATE)

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
        matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return match_img