import cv2
import os
from camera_calibration import undistort
from ball_tracking_yolo import yolo_label_to_pixel_coords
from formulas import dist_formula
from collections import namedtuple

PERSON_LABEL_NUM = '0'
SPORTSBALL_LABEL_NUM = '32'


#source: https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
def intersection_area(bb_shotput, bb_thrower):
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    r_shotput = Rectangle(bb_shotput['x1'], bb_shotput['y1'], bb_shotput['x2'], bb_shotput['y2'])
    r_thrower = Rectangle(bb_thrower['x1'], bb_thrower['y1'], bb_thrower['x2'], bb_thrower['y2'])
    dx = min(r_shotput.xmax, r_thrower.xmax) - max(r_shotput.xmin, r_thrower.xmin)
    dy = min(r_shotput.ymax, r_thrower.ymax) - max(r_shotput.ymin, r_thrower.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else: return 0


# if a sports ball and person are both detected within the search window
# AND sports ball bounding box does not overlap too closely or is not completely within human bounding box
# then that is starting point for shot put flight
def detect_flight(video, box_coords, isLeft):

    #initialize key variables
    shotput_in_flight = False

    #we know coordinates of box upon calibration in the video, create search window above box for ball
    search_window_top_left = (box_coords[0]-350, box_coords[1]-750)
    search_window_bottom_right = (box_coords[0]+350, box_coords[1]+100)
    frame_num = 0

    #read first frame in video
    ret, frame = video.read()
    #looping through frames until shot put is detected to be in flight or video ends
    while shotput_in_flight is False and ret is True:
        #first, undistort image based on camera calibration
        if isLeft:
            frame = undistort(frame, True)
        else:
            frame = undistort(frame, False)


        #write frame so yolo detect can find it
        cv2.imwrite("currentdetectflightframe.png", frame)

        #call my adapted yolov3 detect script on current frame using pretrained yolov3 weights (which is the default, hence why --weights specifier is not included)
        os.system("python3.7 /Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/detect.py"
                  " --source /Users/ashley20/PycharmProjects/ThesisCameraCalibration/currentdetectflightframe.png"
                  " --save-txt"
                  " --save-conf"
                  " --view-img"
                  " --exist-ok"
                  )

        #TESTING CODE
        testframe = cv2.imread('/Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/runs/detect/exp/currentdetectflightframe.png')
        cv2.rectangle(testframe, search_window_top_left, search_window_bottom_right,
                      (255, 0, 0), 5)
        cv2.imshow("test", testframe)
        cv2.waitKey(0)

        # open txt file of labels that yolov3 creates during detect.py
        with open(
                '/Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/runs/detect/exp/labels/currentdetectflightframe.txt',
                'r') as label_file:
            #initialize dictionary used to process detected object information returned from label file
            #key = class label number, will either be '0' or '32'
            #value = array of pixel values for every detected object in the respective category
            detected_objs = {}

            #loop through label file to process detected objects
            for line in label_file:
                line = line.split()
                object_label_num = line[0]
                #if the current detected object is a person or sports ball, add to dictionary
                #else ignore object
                if object_label_num == PERSON_LABEL_NUM or object_label_num == SPORTSBALL_LABEL_NUM:
                    x, y, width, height, confidence = yolo_label_to_pixel_coords(line, frame)
                    #adding information to dictionary
                    if object_label_num in detected_objs:
                        objs = detected_objs[object_label_num]
                        objs.append([x,y,width,height,confidence])
                        detected_objs[object_label_num] = objs
                    else:
                        detected_objs[object_label_num] = [[x,y,width,height,confidence]]

            #check if at least one human and at least one sports ball detected
            #else, move to next frame
            if PERSON_LABEL_NUM in detected_objs and SPORTSBALL_LABEL_NUM in detected_objs:
                #check if at least one sports ball detected is within search window
                shotput_candidates = detected_objs[SPORTSBALL_LABEL_NUM]

                #new array for only shotput candidates within search window
                filtered_shotput_candidates = []
                for sc in shotput_candidates:
                    if (sc[0] > search_window_top_left[0] and sc[0] < search_window_bottom_right[0] and sc[1] > search_window_top_left[1] and sc[1] < search_window_bottom_right[1]):
                        filtered_shotput_candidates.append(sc)

                #if there are shotput candidates inside the search window
                #else, move to next frame
                if len(filtered_shotput_candidates) > 0:
                    #check if at least one person detected is within search window
                    thrower_candidates = detected_objs[PERSON_LABEL_NUM]

                    # new array for only thrower candidates within search window
                    filtered_thrower_candidates = []
                    for tc in thrower_candidates:
                        if (tc[0] > search_window_top_left[0] and tc[0] < search_window_bottom_right[0] and tc[1] > search_window_top_left[1] and tc[1] < search_window_bottom_right[1]):
                            filtered_thrower_candidates.append(tc)

                    # if there are thrower candidates inside the search window, then there is at least one shot put candidate
                    # AND at least one thrower candidate
                    # else, move to next frame
                    if len(filtered_thrower_candidates) > 0:

                        #for each shot put candidate, find person who is the closest by Euclidean distance
                        for sc in filtered_shotput_candidates:
                            all_dists = {}
                            for tc in filtered_thrower_candidates:
                                dist = dist_formula((sc[0], sc[1]), (tc[0], tc[1]))
                                all_dists[(tc[0], tc[1], tc[2], tc[3])] = dist
                            min_key = min(all_dists, key=all_dists.get)
                            min_dist = all_dists[min_key]
                            tc=min_key

                            #create dictionary to construct bounding box of current shot put and thrower candidates
                            x_min = sc[0] - int(sc[2] / 2)
                            y_min = sc[1] - int(sc[3] / 2)
                            x_max = sc[0] + int(sc[2] / 2)
                            y_max = sc[1] + int(sc[3] / 2)
                            bb_shotput = {
                                'x1': x_min,
                                'x2': x_max,
                                'y1': y_min,
                                'y2': y_max
                            }
                            x_min = tc[0] - int(tc[2] / 2)
                            y_min = tc[1] - int(tc[3] / 2)
                            x_max = tc[0] + int(tc[2] / 2)
                            y_max = tc[1] + int(tc[3] / 2)
                            bb_thrower = {
                                'x1': x_min,
                                'x2': x_max,
                                'y1': y_min,
                                'y2': y_max
                            }

                            #if shot put bounding box is entirely inside thrower bounding box, then ignore and move to next frame
                            #this is to prevent the shot put from being detected while it is still in the thrower's hand
                            if bb_shotput['x1'] > bb_thrower['x1'] and bb_shotput['y1'] > bb_thrower['y1'] and bb_shotput['x2'] < bb_thrower['x2'] and bb_shotput['y2'] < bb_thrower['y2']:
                                print("nested bounding boxes")
                                continue

                            #calculate percent area of shot put bounding box that is within thrower bounding box
                            #it must be below a certain threshold to be deemed "has left thrower's hand"
                            intersected_area = intersection_area(bb_shotput, bb_thrower)
                            bb_shotput_area = int(sc[2])*int(sc[3])
                            area_ratio = intersected_area/bb_shotput_area

                            #check that shot put is close enough to thrower without overlapping too much
                            #if these criteria are true, then shot put has just left the thrower's hand
                            if min_dist < 500 and area_ratio < 0.6:
                                shotput_in_flight = True
                                print(min_dist)
                                print(area_ratio)
                                #TESTING CODE
                                # cv2.rectangle(testframe, search_window_top_left, search_window_bottom_right,
                                #               (255, 0, 0), 5)
                                # cv2.circle(testframe, (bb_thrower['x1'], bb_thrower['y1']),5, (255, 255, 0), 5)
                                # cv2.circle(testframe, (bb_thrower['x2'], bb_thrower['y2']), 5, (255, 255, 0), 5)
                                # cv2.imshow('f', testframe)
                                # cv2.waitKey(0)
                                return (int(sc[0]), int(sc[1]))

        #increments frame number
        frame_num += 1
        #gets next frame
        ret, frame = video.read()

    #if video ends and while loop exits, but never finds the flight of shot put
    if shotput_in_flight is False:
        print("video ended before detecting shot put in flight")
        return None

if __name__ == '__main__':
   print("in detect flight")
   video = cv2.VideoCapture('Thesis_Data_Videos_Test/throwfar_1_24_behind_shot_on_190_left.MP4')
   box_coords_left = (1050, 1200)
   box_coords_right = (1005, 1250)
   if not video.isOpened():
       print("no video opened")
       exit()
   detect_flight(video, box_coords_left, isLeft=True)
