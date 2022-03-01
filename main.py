#Regular Camera Calibration for two GoPro Hero7 cameras, Left and Right
#Code adapted from https://www.theeminentcodfish.com/gopro-calibration/

import numpy as np
import cv2

from scene_calibration import click_scene_coordinates
from detect_flight import detect_flight
from ball_tracking_yolo import track_shot_put
from camera_calibration import undistort


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #STEP 1 - read videos
    video_left = cv2.VideoCapture('Thesis_Data_Videos_Test/throwclose_2_414_behind_shot_on_190_left.MP4')
    video_right = cv2.VideoCapture('Thesis_Data_Videos_Test/throwclose_2_414_behind_shot_on_190_right.MP4')

    if video_left.isOpened() is False or video_right.isOpened() is False:
        print("Could not read videos, exiting code")
        exit()

    #STEP 2 - read first frame of video
    success_left, init_frame_left = video_left.read()
    success_right, init_frame_right = video_right.read()

    if success_left is False or success_right is False:
        print("Could not read frames for scene calibration, exiting code")
        exit()

    #STEP 3 - calibrate scene
    #input = initial frames
    #output = shot box coordinates, throwing sector boundaries, any measurement necessities
    stop_box_coords_center_left, stop_box_coords_center_right, stop_box_coords_corners_left, stop_box_coords_corners_right, outer_sector_coords_left, outer_sector_coords_right, eight_coords_left, eight_coords_right\
        = click_scene_coordinates(init_frame_left, init_frame_right)
    cv2.destroyWindow('current frame')


    #STEP 4 - get coordinates when shot put takes flight
    init_shot_put_coords_left = detect_flight(video_left, stop_box_coords_center_left[0], outer_sector_coords_left, isLeft=True)
    init_shot_put_coords_right = detect_flight(video_right, stop_box_coords_center_right[0], outer_sector_coords_right, isLeft=False)

    #STEP 5 - track shot put to get landing coordinates
    #input = initial shot put coordinates of flight, frame number where flight starts
    #output = landing point in frame coordinates
    # _, frame = video_left.read()
    # frame = undistort(frame, isLeft=True)
    # cv2.circle(frame, init_shot_put_coords_left, 5, (255,0,0), 5)
    # cv2.imshow("f", frame)
    # cv2.waitKey(0)
    # _, frame = video_right.read()
    # frame = undistort(frame, isLeft=False)
    # cv2.circle(frame, init_shot_put_coords_right, 5, (255, 0, 0), 5)
    # cv2.imshow("f", frame)
    # cv2.waitKey(0)

    landing_point_left = track_shot_put(video_left, init_shot_put_coords_left, is_left=True)
    landing_point_right = track_shot_put(video_right, init_shot_put_coords_right, is_left=False)

    _, frame = video_left.read()
    frame = undistort(frame, isLeft=True)
    cv2.circle(frame, landing_point_left, 5, (255,0,0), 5)
    cv2.imshow("f", frame)
    cv2.waitKey(0)
    _, frame = video_right.read()
    frame = undistort(frame, isLeft=False)
    cv2.circle(frame, landing_point_right, 5, (255, 0, 0), 5)
    cv2.imshow("f", frame)
    cv2.waitKey(0)

    #STEP 6 - measure shot put
    #input = landing point, stop box coordinates from calibration
    #output = measurement




    




