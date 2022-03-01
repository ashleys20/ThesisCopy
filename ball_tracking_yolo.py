#drawing path and tracking ball based on yolo detection and projectile math
#based on heuristic proposed in vball-net

#loop through frames of video (start at ideal frame)
#get ball candidate at frame, save ball candidates
#get potential paths, save potential paths
#narrow down to find ideal path

import cv2
import math
from camera_calibration import undistort
import os
from formulas import dist_formula

#ball information
#key = frame number
#value = {
#     pred_pos_x
#     # pred_pos_y
#     det_pos_x
#     det_pos_y
#     is_pred
#     vel_x
#     vel_y
#     accel_x
#     accel_y
#     vel
#     accel
#     angle
# }

#helper function to convert line of yolo label (written as percentages) to pixel coordinates
def yolo_label_to_pixel_coords(line, frame):
    h, w, _ = frame.shape
    x = int(float(line[1]) * w)
    y = int(float(line[2]) * h)
    width = float(line[3]) * w
    height = float(line[4]) * h
    confidence = float(line[5])
    return x,y,width,height, confidence

#helper function to predict position of shot put in subsequent frame based on info from current frame
def predict_position(pos, vel, accel):
    return pos + vel + 0.5*accel

#helper function to determine how many frames since last detected point
def get_time_steps(ball_info, flight_count):
    fn = flight_count - 1
    while ball_info[fn]['is_pred'] is True: fn -= 1
    return flight_count - fn

#helper function to determine a threshold based on following criteria:
# 1. threshold based on avg. of width and height of bounding box in the last frame that took the detected point
    #--> meaning if the shot put is bigger w.r.t. frame, the search window will be larger and the same applies to smaller settings
# 2. threshold increases as number of consecutive predicted points grows
    #--> allows room for error when interpolating predicted point
def calc_threshold(ball_info, flight_count):
    #set initial threshold in case doesn't pass checks
    threshold=100

    # get previous bounding box width and height from last frame that took detected point
    fn = flight_count - 1
    while ball_info[fn]['is_pred'] is True: fn -= 1
    curr_width = ball_info[fn]['bb_width']
    curr_height = ball_info[fn]['bb_height']

    # just extra check, should always have values here
    if curr_width is not None and curr_height is not None:
        avg_dimension = (curr_height + curr_width) / 2
        # set threshold based on bounding box size
        threshold = avg_dimension * 5
    # print("threshold: ", threshold)

    # get number of frames since shot put has been accurately detected
    fn = flight_count
    while ball_info[fn - 1]['is_pred'] is True: fn -= 1
    time_steps = flight_count - fn

    # increasing threshold by 20% for each time point is predicted
    for x in range(time_steps):
        threshold *= 1.1

    return threshold

#helper functions to recalculate velocity and acceleration values to be more closely related to the detected points
def recalc_vel_x(ball_info, flight_count, x):
    time_steps = get_time_steps(ball_info, flight_count)
    return (2 * (x - ball_info[flight_count-time_steps]['det_pos_x']) / time_steps - ball_info[flight_count-time_steps]['vel_x']) if (time_steps - ball_info[flight_count-time_steps]['vel_x']) != 0 else 0
def recalc_vel_y(ball_info, flight_count, y):
    time_steps = get_time_steps(ball_info, flight_count)
    return (2 * (y - ball_info[flight_count - time_steps]['det_pos_y']) / time_steps - ball_info[flight_count - time_steps][
        'vel_y']) if (time_steps - ball_info[flight_count - time_steps]['vel_y']) !=0 else 0
def recalc_accel_x(ball_info, flight_count, x, vel_x):
    time_steps = get_time_steps(ball_info, flight_count)
    return (math.pow(vel_x, 2) - math.pow(ball_info[flight_count - time_steps]['vel_x'], 2)) / (2 * (x - ball_info[flight_count - time_steps]['det_pos_x'])) if (2 * (x - ball_info[flight_count - time_steps]['det_pos_x'])) != 0 else 0
def recalc_accel_y(ball_info, flight_count, y, vel_y):
    time_steps = get_time_steps(ball_info, flight_count)
    return (math.pow(vel_y, 2) - math.pow(ball_info[flight_count - time_steps]['vel_y'], 2)) / (2 * (y - ball_info[flight_count - time_steps]['det_pos_y'])) if (2 * (y - ball_info[flight_count - time_steps]['det_pos_y'])) != 0 else 0


def track_shot_put(video, init_shot_put_coords, is_left):

    #initialize ball tracking dictionary, frame number, read first frame, video for writing purposes
    ball_info = {}
    has_landed = False
    success, frame = video.read()
    out = cv2.VideoWriter('ball_tracking_292_conf_thresh_1/2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                          (frame.shape[1], frame.shape[0]))
    # undistorting image based on camera calibration
    frame = undistort(frame, is_left)

    # write image so detect.py can find it
    cv2.imwrite("current_frame.png", frame)

    cv2.circle(frame, (init_shot_put_coords[0], init_shot_put_coords[1]), 5, (255,0,255), 5)
    cv2.imshow("c", frame)
    cv2.waitKey(0)


    #initialize flight count to keep track of how long shot put in air, initalize shot put at given frame, coordinates from detect flight
    flight_count = 0
    landed_count = 0
    ball_info[flight_count] = {
        'det_pos_x': init_shot_put_coords[0],
        'det_pos_y': init_shot_put_coords[1],
        'vel_x': 0,
        'vel_y': 0,
        'accel_x': 0,
        'accel_y': 0,
        'is_pred': False,
        'bb_width': init_shot_put_coords[2],
        'bb_height': init_shot_put_coords[3]
    }
    flight_count+=1
    ball_info[flight_count] = {
        'pred_pos_x': init_shot_put_coords[0],
        'pred_pos_y': init_shot_put_coords[1]
    }

    # loop through rest of frames in video for tracking
    while success is True and frame is not None:
        print("flight count = ")
        print(flight_count)
        if has_landed is True: landed_count+=1

        cv2.imwrite("testing_current_frame.png", frame)

        cv2.imwrite("current_frame.png", frame)

        #shortcut naming --> set current frame in dictionary to b
        b = ball_info[flight_count]
        print(ball_info)

        # different tests on detect:
        # time-accuracy tradeoff between running detect on img size of 640 vs 1280?
        # time-accuracy tradeoff of running detect on search window rather than full image
        # time-accuracy tradeoff of running detect on serach window that is super-resolutioned rather than full, un-resolutioned image

        #call detect.py using weights from custom trained yolov3 model
        os.system("python3.7 /Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/detect.py"
                  " --weights '/Users/ashley20/PycharmProjects/ThesisCameraCalibration/last_aug_final.pt'"
                  " --source /Users/ashley20/PycharmProjects/ThesisCameraCalibration/current_frame.png"
                  " --save-txt"
                  " --save-conf"
                  " --exist-ok"
                  )


        #open txt file of labels that yolov3 creates during detect.py
        with open('/Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/runs/detect/exp/labels/current_frame.txt','r') as label_file:
            #getting number of detections in current frame
            with  open('/Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/runs/detect/exp/labels/current_frame_num_detections.txt','r') as num_detections_file:
                for line in num_detections_file:
                    line=line.split()
                    num_detections = int(line[0])
            print("num detections: ", num_detections)
            #if label file is empty, this means no shot put candidates, so prediction automatically becomes point
            if num_detections == 0:
                print("no detections found")
                x = b['pred_pos_x']
                y = b['pred_pos_y']
                b['det_pos_x'] = x
                b['det_pos_y'] = y
                prev_info = ball_info[flight_count - 1]
                vel_x = x - prev_info['det_pos_x']
                vel_y = y - prev_info['det_pos_y']
                b['vel_x'] = vel_x
                b['vel_y'] = vel_y
                b['vel'] = math.sqrt(vel_x ** 2 + vel_y ** 2)
                b['angle'] = math.atan(vel_y / vel_x) if vel_x != 0 else float("-inf")
                if flight_count > 1:
                    accel_x = vel_x - prev_info['vel_x']
                    accel_y = vel_y - prev_info['vel_y']
                else:
                    accel_x = 0
                    accel_y = 0
                b['accel_x'] = accel_x
                b['accel_y'] = accel_y
                b['accel'] = math.sqrt(accel_x ** 2 + accel_y ** 2)
                b['is_pred'] = True
                b['bb_width'] = None
                b['bb_height'] = None

                # now, finally have enough info to make prediction about next frame with all values
                ball_info[flight_count+1] = {}
                ball_info[flight_count + 1]['pred_pos_x'] = predict_position(x, vel_x, accel_x)
                ball_info[flight_count + 1]['pred_pos_y'] = predict_position(y, vel_y, accel_y)


                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), 5)

            #else if there are ball candidates
            else:
                #first, collect all shot put candidates, then find point with shortest distance from previous point
                #then, create search window from predicted ball position and make sure point falls within search window
                pred_ball_x = b['pred_pos_x']
                pred_ball_y = b['pred_pos_y']

                #code to find best shot put candidate
                all_dists = {}
                for line in label_file:
                    line = line.split()
                    x,y,width,height, confidence = yolo_label_to_pixel_coords(line, frame)
                    dist = dist_formula((x,y), (pred_ball_x, pred_ball_y))
                    score = dist +10*(1-confidence)
                    print("dist = ", dist)
                    print("conf = ", confidence)
                    print("score = ", score)
                    all_dists[(x,y)] = score
                min_key = min(all_dists, key=all_dists.get)

                #if min distance falls within threshold, take it as the ball; if not, then take the prediction
                #adjusting threshold based on two criteria:

                # set initial threshold
                threshold = calc_threshold(ball_info, flight_count)

                #if distance falls within threshold, take it as the shot put point
                if all_dists[min_key] < threshold:
                    b['det_pos_x'] = min_key[0]
                    b['det_pos_y'] = min_key[1]
                    b['bb_width'] = width
                    b['bb_height'] = height
                    b['is_pred'] = False
                #else, take predicted point
                else:
                    b['det_pos_x'] = b['pred_pos_x']
                    b['det_pos_y'] = b['pred_pos_y']
                    b['is_pred'] = True
                    #if predicted point, there is no known bounding box
                    b['bb_width'] = None
                    b['bb_height'] = None

                cv2.circle(frame, (int(b['det_pos_x']), int(b['det_pos_y'])), 5, (0, 255, 0), 5)

                if flight_count == 1:
                    b['is_pred'] = False
                    b['vel_x'] = 0
                    b['vel_y'] = 0
                    b['accel_x'] = 0
                    b['accel_y'] = 0
                    b['vel'] = 0
                    b['accel'] = 0

                    #if first frame, then assume vel and accel are negligible, so position is same place
                    ball_info[flight_count + 1] = {}
                    ball_info[flight_count + 1]['pred_pos_x'] = x
                    ball_info[flight_count + 1]['pred_pos_y'] = y

                if flight_count >= 1:
                    prev_info = ball_info[flight_count-1]

                    #reset velocity values if back to predicted, calculate them based on last detected values
                    if (prev_info['is_pred'] is True) and (b['is_pred'] is False):
                        vel_x = recalc_vel_x(ball_info, flight_count, x)
                        vel_y = recalc_vel_y(ball_info, flight_count, y)
                    #if not, take normal change in position for velocity
                    else:
                        vel_x = x - prev_info['det_pos_x']
                        vel_y = y - prev_info['det_pos_y']

                    #set values in dictionary
                    b['vel_x'] = vel_x
                    b['vel_y'] = vel_y
                    b['vel'] = math.sqrt(vel_x**2 + vel_y**2)
                    b['angle'] = math.atan(vel_y / vel_x) if vel_x != 0 else float("-inf")

                    # now, finally have enough info to make prediction about next frame, but assume acceleration here is still negligible
                    ball_info[flight_count + 1] = {}
                    ball_info[flight_count+1]['pred_pos_x'] = predict_position(x, vel_x, 0)
                    ball_info[flight_count+1]['pred_pos_y'] = predict_position(y, vel_y, 0)

                if flight_count >=2:
                    prev_info = ball_info[flight_count - 1]
                    # reset acceleration values if back to predicted, calculate them based on last detected values
                    if (prev_info['is_pred'] is True) and (ball_info[flight_count]['is_pred'] is False):
                        accel_x = recalc_accel_x(ball_info, flight_count, x, vel_x)
                        accel_y = recalc_accel_y(ball_info, flight_count, y, vel_y)
                    #if not, take normal change in velocity for acceleration
                    else:
                        accel_x = vel_x - prev_info['vel_x']
                        accel_y = vel_y - prev_info['vel_y']

                    b['accel_x'] = accel_x
                    b['accel_y'] = accel_y
                    ball_info[flight_count]['accel'] = math.sqrt(accel_x**2 + accel_y**2)

                    # now, finally have enough info to make prediction about next frame with all values
                    ball_info[flight_count + 1]['pred_pos_x'] = predict_position(x, vel_x, accel_x)
                    ball_info[flight_count + 1]['pred_pos_y'] = predict_position(y, vel_y, accel_y)

        #code to incrementally draw points on video
        for key in ball_info:
            if key < flight_count:
                x_pos = int(ball_info[key]['det_pos_x'])
                y_pos = int(ball_info[key]['det_pos_y'])
                if ball_info[key]['is_pred'] is False:
                    cv2.circle(frame, (x_pos,y_pos), 5, (0,255,0), 5)
                else: cv2.circle(frame, (x_pos,y_pos), 5, (0,215,255), 5)
        #if flight_count > 10:
            #cv2.imshow("f", frame)
            #cv2.waitKey(0)

        #code to figure out if shot put has landed
        if flight_count > 2 and has_landed is False:
            print("checking if landed")
            prev_info = ball_info[flight_count - 1]
            curr_v_y = b['vel_y']
            prev_v_y = prev_info['vel_y']
            print(curr_v_y)
            print(prev_v_y)
            print(b['vel_x'])
            print(prev_info['vel_x'])

            #if critical point in y-velocity is found, assume shot put has landed
            if curr_v_y < 0 and prev_v_y > 0:
                cv2.putText(frame, "LANDED!", (50,1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                #cv2.circle(frame, (int(ball_info[flight_count]['det_pos_x']), int(ball_info[flight_count]['det_pos_y'])), 20, (0,0,255), -1)
                has_landed = True
                cv2.imwrite('landed_img.png', frame)
                cv2.imshow("landed frame", frame)
                cv2.waitKey(0)

                #now that landing frame has been found, try to interpolate exact landing pixel coordinate
                #first figure out time step to get from current velocity to 0 velocity
                time_step = abs((0-prev_info['vel'])/prev_info['accel']) if prev_info['accel'] !=0 else 0

                #then apply that time step to position equations to get x and y coordinate prediction
                prev_x = prev_info['det_pos_x']
                prev_y = prev_info['det_pos_y']
                prev_v_x = prev_info['vel_x']
                prev_accel_x = prev_info['accel_x']
                prev_accel_y = prev_info['accel_y']

                pred_landing_x = prev_x + prev_v_x*time_step + 0.5*prev_accel_x*math.pow(time_step, 2)
                pred_landing_y = prev_y + prev_v_y*time_step + 0.5 * prev_accel_y * math.pow(time_step, 2)

                cv2.putText(frame, "Predicted Landing Point: (" + str(pred_landing_x) + ', ' + str(pred_landing_y) + ')', (50, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        #write frame to video, reset next frame, increment flight count and frame number
        cv2.circle(frame, (int(b['det_pos_x']), int(b['det_pos_y'])), 5, (255, 255, 0), 5)
        # if frame_num > 53:
        #     print(b['pred_pos_x'])
        #     print(b['pred_pos_y'])
        #     print(b['det_pos_x'])
        #     print(b['det_pos_y'])
        #     print(b['vel_x'])
        #     print(b['vel_y'])
        #     print(b['accel_x'])
        #     print(b['accel_y'])
        #     cv2.imshow('f', frame)
        #     cv2.waitKey(0)
        out.write(frame)
        ret, frame = video.read()
        frame = undistort(frame, is_left)
        flight_count+=1

    #finish video after while loop exits
    out.release()

if __name__ == '__main__':
   print("in_ball_tracking_yolo")

   #read in left and right videos
   video_left = cv2.VideoCapture('Thesis_Data_Videos_Left/throwfar_2_292_behind_shot_on_160_left.MP4')
   video_right = cv2.VideoCapture('Thesis_Data_Videos_Right/throwfar_2_292_behind_shot_on_160_right.MP4')
   video_left = cv2.VideoCapture('Thesis_Data_Videos_Test/throwclose_2_414_behind_shot_on_190_left.MP4')
   #video_right = cv2.VideoCapture('Thesis_Data_Videos_Test/throwclose_2_414_behind_shot_on_190_right.MP4')
   #video_left = cv2.VideoCapture('Thesis_Data_Videos_Test/throwfar_1_24_behind_shot_on_190_left.MP4')
   #video_right = cv2.VideoCapture('Thesis_Data_Videos_Test/throwfar_1_24_behind_shot_on_190_right.MP4')

   #make sure videos can be opened, then call track_ball
   if not video_left.isOpened() or not video_right.isOpened():
       print("no video opened")
       exit()
   else:
       video = video_left
       for x in range(16):
           success, frame = video.read()
       #cv2.imshow("f", frame)
       #cv2.waitKey(0)
       track_shot_put(video, (1045,852, 29, 27), is_left=True)

