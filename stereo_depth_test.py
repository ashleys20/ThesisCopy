import numpy as np
import cv2
from matplotlib import pyplot as plt
from camera_calibration import undistort

def get_depth(video_left, video_right):

    for i in range(0, 96):
        ret_left, frame_left = video_left.read()
        ret_right, frame_right = video_right.read()

    for i in range(0, 8):
        ret_right, frame_right = video_right.read()
        #cv2.imshow("right", frame_right)
        #cv2.waitKey(0)

    cv2.imshow("left", frame_left)
    cv2.imshow("right", frame_right)
    cv2.waitKey(0)

    frame_left = cv2.cvtColor(undistort(frame_left, True), cv2.COLOR_BGR2GRAY)
    frame_right = cv2.cvtColor(undistort(frame_right, False), cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(frame_left, frame_right)
    plt.imshow(disparity, 'gray')
    plt.show()





if __name__ == '__main__':
   print("in ball tracking yolo")
   video_left = cv2.VideoCapture('Thesis_Data_Videos_Left/throwfar_2_292_behind_shot_on_160_left.MP4')
   video_right = cv2.VideoCapture('Thesis_Data_Videos_Right/throwfar_2_292_behind_shot_on_160_right.MP4')
   if not video_left.isOpened() or not video_right.isOpened():
       print("no video opened")
       exit()
   else: get_depth(video_left, video_right)