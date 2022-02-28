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
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(params[0], str(x) + ',' +
                    str(y), (x, y), font,
                    1, (0, 255, 0), 2)
        cv2.circle(params[0], (x, y), 5, (0, 255, 0), 5)
        cv2.imshow('current frame', params[0])
        params[1].append((x,y))

def click_event_eight_coords(event, x, y, flags, params):
    width = params[3]
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(params[0], str(x) + ',' +
                    str(y), (x, y), font,
                    1, (0, 255, 0), 2)
        cv2.circle(params[0], (x, y), 5, (0, 255, 0), 5)
        #cv2.circle(params[0], (x+params[3], y), 5, (0, 255, 0), 5)
        cv2.imshow('current frame', params[0])

        #adjust depending on if left or right frame, adjust right frame coordinates because of concatenation
        if x < width:
            params[1].append((x,y))
        else:
            x = x - width
            params[2].append((x,y))

def mark_stop_box_center(frame_left, frame_right):
    stop_box_coords_center_left = []
    stop_box_coords_center_right = []

    cv2.putText(frame_left, "First, let's mark the location of the stop box in the left camera.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_left, "Click the center of the stop box in the frame.",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_left, "Then, hit 'Enter' on your keyboard.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow('current frame', frame_left)
    params = [frame_left, stop_box_coords_center_left]

    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.putText(frame_right, "Now, let's mark the location of the stop box in the right camera.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_right, "Click the center of the stop box in the frame.",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_right, "Then, hit 'Enter' on your keyboard.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow('current frame', frame_right)
    params = [frame_right, stop_box_coords_center_right]

    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return stop_box_coords_center_left, stop_box_coords_center_right

def mark_stop_box_corners(frame_left, frame_right):
    stop_box_coords_corners_left = []
    stop_box_coords_corners_right = []
    cv2.putText(frame_left, "Now, let's mark the location of the stop box corners in the left camera.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_left, "In this order, click the upper left, upper right, ",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_left, "bottom left and bottom right corners of the stop box.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_left, "Then, hit 'Enter' on your keyboard.",
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow('current frame', frame_left)
    params = [frame_left, stop_box_coords_corners_left]

    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.putText(frame_right, "Now, let's mark the location of the stop box corners in the right camera.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_right,
                "In this order, click the upper left, upper right, bottom left and bottom right corners of the stop box.",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_right, "Then, hit 'Enter' on your keyboard.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow('current frame', frame_right)
    params = [frame_right, stop_box_coords_corners_right]

    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return stop_box_coords_corners_left, stop_box_coords_corners_right

def mark_throwing_sector(frame_left, frame_right):
    outer_sector_coords_left = []
    outer_sector_coords_right = []

    cv2.putText(frame_left, "Next, let's mark the location of the throwing sector in the left camera.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_left, "In this order, click the far left and far right ",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_left, "corners of the throwing sector.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_left, "Then, hit 'Enter' on your keyboard.",
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow('current frame', frame_left)
    params = [frame_left, outer_sector_coords_left]

    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.putText(frame_right, "Now, let's mark the location of the throwing sector in the right camera.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_right, "In this order, click the far left and far right ",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_right, "corners of the throwing sector.",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame_right, "Then, hit 'Enter' on your keyboard.",
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow('current frame', frame_right)
    params = [frame_right, outer_sector_coords_right]

    cv2.setMouseCallback("current frame", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return outer_sector_coords_left, outer_sector_coords_right

def get_eight_coords(frame_left, frame_right):
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
    cv2.imshow('current frame', combined_frame)
    params = [combined_frame, eight_coords_left, eight_coords_right, width]

    cv2.setMouseCallback("current frame", click_event_eight_coords, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return eight_coords_left, eight_coords_right


def click_scene_coordinates(video_left, video_right):
    success_left, initial_frame_left = video_left.read()
    success_right, initial_frame_right = video_right.read()
    initial_frame_left = undistort(initial_frame_left, isLeft=True)
    initial_frame_right = undistort(initial_frame_right, isLeft=False)

    if success_left is False or success_right is False:
        print("Could not read images for calibration, exiting code")
        exit()
    else:
        #TODO: add error checking for each markings so if there are too many or not enough marked points, will catch that
        #TODO: add ability to re-click if clicked wrong place the first time or unsatisfied with the selection

        #STEP 1 - mark stop box center
        stop_box_coords_center_left, stop_box_coords_center_right = mark_stop_box_center(initial_frame_left.copy(), initial_frame_right.copy())

        #STEP 2 - mark stop box corners
        stop_box_coords_corners_left, stop_box_coords_corners_right = mark_stop_box_corners(initial_frame_left.copy(), initial_frame_right.copy())

        #STEP 3 - mark throwing sector boundaries
        outer_sector_coords_left, outer_sector_coords_right = mark_throwing_sector(initial_frame_left.copy(), initial_frame_right.copy())

        #STEP 4 - gather other coordinates for the 8-points algorithm
        eight_coords_left, eight_coords_right = get_eight_coords(initial_frame_left.copy(), initial_frame_right.copy())


        print(stop_box_coords_center_left)
        print(stop_box_coords_center_right)
        print(stop_box_coords_corners_left)
        print(stop_box_coords_corners_right)
        print(outer_sector_coords_left)
        print(outer_sector_coords_right)
        print(eight_coords_left)
        print(eight_coords_right)

        return stop_box_coords_center_left, stop_box_coords_center_right, stop_box_coords_corners_left, stop_box_coords_corners_right, outer_sector_coords_left, outer_sector_coords_right, eight_coords_left, eight_coords_right





if __name__ == '__main__':
    print("in calibrate scene")
    video_left = cv2.VideoCapture('Thesis_Data_Videos_Test/throwclose_2_414_behind_shot_on_190_left.MP4')
    video_right = cv2.VideoCapture('Thesis_Data_Videos_Test/throwclose_2_414_behind_shot_on_190_right.MP4')
    click_scene_coordinates(video_left, video_right)