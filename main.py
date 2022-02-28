#Regular Camera Calibration for two GoPro Hero7 cameras, Left and Right
#Code adapted from https://www.theeminentcodfish.com/gopro-calibration/

import numpy as np
import cv2

import camera_calibration as calib
import find_implement as find
import object_detection_yolo_test as yolo
import ball_tracking as track
import ball_tracking_projectile_estimation as ball_tracking_proj
from scene_calibration import calibrate_scene
from detect_flight import detect_flight
from ball_tracking_yolo import track_ball_kinematics


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    video_left = cv2.VideoCapture()
    video_right = cv2.VideoCapture()

    success_left, init_frame_left = cv2.imread(video_left)
    success_right, init_frame_right = cv2.imread(video_right)

    #calibrate scene
    #input is initial frames
    #output is shot box coordinates, throwing sector boundaries, any measurement necessities
    shot_box_coords, sector_bounds = calibrate_scene(init_frame_left, init_frame_right)

    init_shot_put_coords_left = detect_flight(video_left, shot_box_coords, isLeft=True)
    init_shot_put_coords_right = detect_flight(video_right, shot_box_coords, isLeft=False)

    landing_point_left = track_ball_kinematics(video_left)
    landing_point_right = track_ball_kinematics(video_right)


    




