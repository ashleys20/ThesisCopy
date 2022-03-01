#calibration of scene at beginning after cameras are set up
import cv2
import numpy as np
from camera_calibration import undistort

#open first frame of left and right video
#click for shot box center and corner coordinates
#click corners of throwing sector
#click other corresponding points

#construct fundamental matrix
#construct essential matrix


def click_event(event, x, y, flags, params):
    width = params[3]
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        if x > width: x_label = str(x-width)
        else: x_label = str(x)
        cv2.putText(params[0], x_label + ',' +
                    str(y), (x, y), font,
                    1, (0, 255, 0), 2)
        cv2.circle(params[0], (x, y), 5, (0, 255, 0), 5)
        #cv2.circle(params[0], (x+params[3], y), 5, (0, 255, 0), 5)
        cv2.imshow('current frame', params[0])

        #adjust depending on if left or right frame, adjust right frame coordinates because of concatenation
        if x < width:
            params[1].append((int(x),int(y)))
        else:
            x = x - width
            params[2].append((int(x),int(y)))

def mark_stop_box_center(frame_left, frame_right):
    stop_box_coords_center_left = []
    stop_box_coords_center_right = []

    _, width, _ = frame_left.shape
    combined_frame = np.concatenate((frame_left, frame_right), axis=1)

    cv2.putText(combined_frame, "First, let's mark the location of the stop box in the left camera.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame, "Click the center of the stop box in the frame.",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame, "Then, hit 'Enter' on your keyboard.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.namedWindow('current frame')
    cv2.moveWindow('current frame', 1275, 50)
    cv2.imshow('current frame', combined_frame)

    params = [combined_frame, stop_box_coords_center_left, stop_box_coords_center_right, width]
    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return stop_box_coords_center_left, stop_box_coords_center_right

def mark_stop_box_corners(frame_left, frame_right):
    stop_box_coords_corners_left = []
    stop_box_coords_corners_right = []

    _, width, _ = frame_left.shape
    combined_frame = np.concatenate((frame_left, frame_right), axis=1)

    cv2.putText(combined_frame, "Now, let's mark the location of the stop box corners in each camera.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame, "In this order, click the upper left, upper right, bottom left and bottom right corners of the stop box in each camera.",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame, "Hit 'Enter' on your keyboard when done.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.namedWindow('current frame')
    cv2.moveWindow('current frame', 1275, 50)
    cv2.imshow('current frame', combined_frame)

    params = [combined_frame, stop_box_coords_corners_left, stop_box_coords_corners_right, width]
    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return stop_box_coords_corners_left, stop_box_coords_corners_right

def mark_throwing_sector(frame_left, frame_right):
    outer_sector_coords_left = []
    outer_sector_coords_right = []

    _, width, _ = frame_left.shape
    combined_frame = np.concatenate((frame_left, frame_right), axis=1)

    cv2.putText(combined_frame, "Next, let's mark the location of the throwing sector in each camera.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame, "First, click the far left and far right corners of the throwing sector in the left image.",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame, "Then, do the same in the right image.  Hit 'Enter' on your keyboard when done.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.namedWindow('current frame')
    cv2.moveWindow('current frame', 1275, 50)
    cv2.imshow('current frame', combined_frame)

    params = [combined_frame, outer_sector_coords_left, outer_sector_coords_right, width]
    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return outer_sector_coords_left, outer_sector_coords_right

def mark_eight_coords(frame_left, frame_right):
    eight_coords_left = []
    eight_coords_right = []
    _, width, _ = frame_left.shape
    combined_frame = np.concatenate((frame_left, frame_right), axis=1)

    cv2.putText(combined_frame,
                "Finally, let's mark the location of corresponding coordinates in the left and right cameras.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame,
                "Click a point in the left frame and click its corresponding point in the right frame, in that order.",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame,
                "Alternate clicking in points in left and right frames until you reach 8 pairs of coordinates.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame, "Try and click points that cover a wide range of locations in the image.",
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(combined_frame, "Then, hit 'Enter' on your keyboard.",
                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.namedWindow('current frame')
    cv2.moveWindow('current frame', 1400, 50)
    cv2.imshow('current frame', combined_frame)
    params = [combined_frame, eight_coords_left, eight_coords_right, width]

    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return eight_coords_left, eight_coords_right


def click_scene_coordinates(initial_frame_left, initial_frame_right):
    initial_frame_left = undistort(initial_frame_left, isLeft=True)
    initial_frame_right = undistort(initial_frame_right, isLeft=False)

    #TODO: add error checking for each markings so if there are too many or not enough marked points, will catch that
    #TODO: add ability to re-click if clicked wrong place the first time or unsatisfied with the selection

    #STEP 1 - mark stop box center
    stop_box_coords_center_left, stop_box_coords_center_right = mark_stop_box_center(initial_frame_left.copy(), initial_frame_right.copy())

    #STEP 2 - mark stop box corners
    stop_box_coords_corners_left, stop_box_coords_corners_right = mark_stop_box_corners(initial_frame_left.copy(), initial_frame_right.copy())

    #STEP 3 - mark throwing sector boundaries
    outer_sector_coords_left, outer_sector_coords_right = mark_throwing_sector(initial_frame_left.copy(), initial_frame_right.copy())

    #STEP 4 - gather other coordinates for the 8-points algorithm
    eight_coords_left, eight_coords_right = mark_eight_coords(initial_frame_left.copy(), initial_frame_right.copy())


    #TESTING CODE
    # for point in stop_box_coords_center_right:
    #     cv2.circle(initial_frame_right, point, 5, (0,255,0), 5)
    # for point in stop_box_coords_corners_right:
    #     cv2.circle(initial_frame_right, point, 5, (0, 255, 0), 5)
    # for point in outer_sector_coords_right:
    #     cv2.circle(initial_frame_right, point, 5, (0, 255, 0), 5)
    # for point in eight_coords_right:
    #     cv2.circle(initial_frame_right, point, 5, (0, 255, 0), 5)
    # cv2.imshow("initial frame with all points", initial_frame_right)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # print(stop_box_coords_center_left)
    # print(stop_box_coords_center_right)
    # print(stop_box_coords_corners_left)
    # print(stop_box_coords_corners_right)
    # print(outer_sector_coords_left)
    # print(outer_sector_coords_right)
    # print(eight_coords_left)
    # print(eight_coords_right)

    cv2.destroyAllWindows()
    return stop_box_coords_center_left, stop_box_coords_center_right, stop_box_coords_corners_left, stop_box_coords_corners_right, outer_sector_coords_left, outer_sector_coords_right, eight_coords_left, eight_coords_right

if __name__ == '__main__':
    print("in calibrate scene")
    video_left = cv2.VideoCapture('Thesis_Data_Videos_Test/throwclose_2_414_behind_shot_on_190_left.MP4')
    video_right = cv2.VideoCapture('Thesis_Data_Videos_Test/throwclose_2_414_behind_shot_on_190_right.MP4')
    success_left, initial_frame_left = video_left.read()
    success_right, initial_frame_right = video_right.read()
    click_scene_coordinates(initial_frame_left, initial_frame_right)