#Regular Camera Calibration for two GoPro Hero7 cameras, Left and Right
#Code adapted from https://www.theeminentcodfish.com/gopro-calibration/

import numpy as np
import cv2

import camera_calibration as calib
import find_implement as find
import object_detection_yolo_test as yolo
import ball_tracking as track
import ball_tracking_projectile_estimation as ball_tracking_proj


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ball_tracking_proj.tracking_left()
    #ball_tracking_proj.tracking_right()
    exit()

    #track.background_subtraction()
    #exit()

    #track.track_shot_put("CSRT")


    #instrinsic calibration for left camera
    #calib.ImageProcessing(15, 9, 6, 25, True)
    #intrinsic calibration for right camera
    #calib.ImageProcessing(15, 9, 6, 25, False)

    has_both_frames = True
    #read first frame from left video
    vs = cv2.VideoCapture('Thesis_Data_Videos_Left/center_1_24_behind_spike_on_190_left.MP4')
    ret, frame_left = vs.read()
    if ret is False: has_both_frames = False

    #read first frame from right video
    vs = cv2.VideoCapture('Thesis_Data_Videos_Right/center_1_24_behind_spike_on_190_right.MP4')
    ret, frame_right = vs.read()
    if ret is False: has_both_frames = False

    if not has_both_frames:
        print("did not read frames")
        exit()


    #before you look to detect, call undistort function here to undistort images based on prev camera calib
    frame_left_undist = calib.undistort(frame_left, isLeft = True)
    frame_right_undist = calib.undistort(frame_right, isLeft=False)
    cv2.imwrite("frame_left_undist.png", frame_left_undist)
    cv2.imwrite("frame_right_undist.png", frame_right_undist)



    # cropping frames
    frame_left_undist = frame_left_undist[0:3000, 600:1300]
    frame_right_undist = frame_right_undist[0:3000, 400:1300]
    cv2.imshow("window", frame_left_undist)
    cv2.waitKey(0)
    cv2.imshow("window", frame_right_undist)
    cv2.waitKey(0)

    if has_both_frames:
        find.find_stop_board(frame_left_undist)
        circle_left, circle_right = find.detection(frame_left_undist, frame_right_undist)
    else:
        print("both frames not read")
        exit()

    # after shot and lines have been detected, check if out of bounds, verify this is the case on both frames
    is_out_of_bounds_left = find.out_of_bounds(frame_left, circle_left)
    is_out_of_bounds_right = find.out_of_bounds(frame_right, circle_right)
    if is_out_of_bounds_left and is_out_of_bounds_right:
        print("OUT OF BOUNDS FOUL")
        exit()
    elif is_out_of_bounds_right != is_out_of_bounds_left:
        print("something wrong, out of bounds detection does not match in both frames")
        exit()
    else:
        print("IN BOUNDS")

    cv2.imshow("window", frame_left)
    cv2.waitKey(0)
    cv2.imshow("window", frame_right)
    cv2.waitKey(0)

