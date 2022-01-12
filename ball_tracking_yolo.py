#drawing path and tracking ball based on yolo detection and projectile math
#based on heuristic proposed in vball-net

#loop through frames of video (start at ideal frame)
#get ball candidate at frame, save ball candidates
#get potential paths, save potential paths
#narrow down to find ideal path

import numpy as np
import cv2
import math
import operator
from camera_calibration import undistort
#from yolov3 import detect
import os
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from ball_tracking_projectile_estimation import dist_formula

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

def track_ball(video):

    # ransac_ball_candidates = [[0.641927, 0.50625], [0.635417, 0.488194], [0.27474, 0.744444], [0.629167, 0.473264], [0.27474, 0.744444], [0.623698, 0.460417], [0.27474, 0.744444], [0.61849, 0.449306], [0.27474, 0.744792], [0.613802, 0.442708], [0.27474, 0.745139], [0.609375, 0.436111], [0.605208, 0.431597]]
    # np_ball_coords = np.array(ransac_ball_candidates)
    # x_vals = []
    # y_vals = []
    # for row in np_ball_coords:
    #     print(row)
    #     x_vals.append(row[0])
    #     y_vals.append(row[1])
    # x_vals = np.array(x_vals)
    # y_vals = np.array(y_vals)
    #
    # model = np.poly1d(np.polyfit(x_vals, y_vals, 2))
    # print(model)
    #
    # exit()


    #ransac = RANSACRegressor(PolynomialFeatures(degree=2), random_state=0)
    #reg = RANSACRegressor.fit(ransac_ball_candidates)
    #reg = RANSACRegressor.fit(X=ransac_ball_candidates[0:5], y=ransac_ball_candidates[5:])

    ball_candidates = {}
    regression_ball_candidates = []

    paths=[]

    frame_num = 0
    ret, frame = video.read()
    # undistorting image based on camera calibration
    frame = undistort(frame, True)

    out = cv2.VideoWriter('ball_candidates_each_frame_kinematics.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                          (frame.shape[1], frame.shape[0]))

    #looping through for now to get to ideal frame
    for i in range(0, 63):
        ret, frame = video.read()
        frame_num +=1


    #loop through rest of frames for tracking
    while frame_num < 115:
        print("in frame loop")
        cv2.imshow('f', frame)
        cv2.waitKey(0)
        #write image so detect.py can find it
        cv2.imwrite("current_frame.png", frame)

        #code to try and zoom in but this actually resulted in more false positives
        #if there are ball candidates from prev frame
        # if frame_num-1 in ball_candidates.keys() and bool(ball_candidates[frame_num-1]):
        #     ball_candidate = ball_candidates[frame_num-1][0]
        #     dh, dw, _ = frame.shape
        #
        #     #center point of ball candidate
        #     x = int(float(ball_candidate[0]) * dw)
        #     y = int(float(ball_candidate[1]) * dh)
        #
        #     cv2.circle(frame, (x,y), 5, (0,0,0), 5)
            #cv2.imshow('orig', frame)
            #frame = frame[y-200:y+200, x-200:x+200]
            #cv2.imshow("f", frame)
            #cv2.waitKey(0)
            #cv2.imwrite('current_frame.png', frame)

        #call detect from yolo on frame
        #always writes to exp, saves txt under exp/labels
        os.system("python3.7 /Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/detect.py"
                  " --weights '/Users/ashley20/Documents/last_neg.pt'"
                  " --source /Users/ashley20/PycharmProjects/ThesisCameraCalibration/current_frame.png"
                  " --save-txt"
                  " --exist-ok")

        ball_candidates[frame_num] = {}
        #pull resulting labels of ball candidates from detect
        with open('/Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/runs/detect/exp/labels/current_frame.txt',
                  'r') as label_file:
            #adding identifier for each ball candidate in the frame
            label_num = 0
            for line in label_file:
                line = line.split()
                x = float(line[1])
                y = float(line[2])
                width = float(line[3])
                height = float(line[4])
                ball_candidates[frame_num][label_num] = [x,y,width,height]
                label_num+=1
                dh, dw, _ = frame.shape
                x = int(float(x) * dw)
                y = int(float(y) * dh)
                regression_ball_candidates.append([x,y])
                cv2.circle(frame, (x,y), 5, (0,255,0), 5)
            out.write(frame)

        #cv2.imshow("f", frame)
        #cv2.waitKey(0)

        if len(paths) == 0:
            for c in ball_candidates[frame_num]:
                print(c)
                #store array of frame_num + label_num so can go back and find it in ball_candidate dictionary
                paths.append([str(frame_num)+"_"+str(c)])
                print(paths)

        else:
            for curr in ball_candidates[frame_num]:
                for path in paths:
                    #print(ball_candidates[frame_num])
                    x_curr, y_curr, w_curr, h_curr = ball_candidates[frame_num][curr][0], ball_candidates[frame_num][curr][1], ball_candidates[frame_num][curr][2], ball_candidates[frame_num][curr][2]
                    prev_frame_num, prev_label_num = path[0].split('_')
                    prev_frame_num = int(prev_frame_num)
                    prev_label_num = int(prev_label_num)
                    x_prev, y_prev, w_prev, h_prev = ball_candidates[prev_frame_num][prev_label_num][0], ball_candidates[prev_frame_num][prev_label_num][1], ball_candidates[prev_frame_num][prev_label_num][2], ball_candidates[prev_frame_num][prev_label_num][2]
                    print(x_curr)
                    print(x_prev)
                    dist = dist_formula([x_curr, y_curr], [x_prev, y_prev])
                    direction = get_direction(ball_candidates[frame_num][curr], ball_candidates[prev_frame_num][prev_label_num])
                    print("math info")
                    print(frame_num, curr)
                    print(frame_num-1, path)
                    print(dist)
                    print(direction)
                    if dist > 0 and dist < 0.05:
                        print("storing path")
                        updated = str(frame_num) + '_' + str(curr)
                        print(updated)
                        path.insert(0, updated)
                        print(path)

        for path in paths:
            print("path:")
            print(path)
            for i in range(len(path)):
                curr_frame_num, curr_label_num = path[i].split('_')
                dict = ball_candidates[int(curr_frame_num)][int(curr_label_num)]
                dh, dw, _ = frame.shape
                x = int(float(dict[0]) * dw)
                y = int(float(dict[1]) * dh)
                width = int(float(dict[2]) * dw)
                height = int(float(dict[3]) * dh)
                x_max = x + int(width / 2)
                y_max = y + int(height / 2)
                cv2.circle(frame, (x,y), 5, (0,0,255), 5)
                #regression_ball_candidates.append([x,y])

        for all_bc in regression_ball_candidates:
            cv2.circle(frame, (all_bc[0], all_bc[1]), 5, (0, 255, 0), 5)


        #out.write(frame)
        if frame_num == 96:
            cv2.imshow("frame", frame)
            cv2.imwrite("all_ball_candidates_detector_only.png", frame)
            cv2.waitKey(0)

        #go to next frame
        ret, frame = video.read()
        frame_num+=1

    #print(ransac_ball_candidates)
    print(ball_candidates)
    print(paths)
    out.release()

    "print long paths"
    for path in paths:
        if len(path) > 3:
            print(path)



def get_direction(curr, prev):
    return [curr[0] - prev[0], curr[1]-prev[1]]


def track_ball_kinematics(video):
    ball_info = {}

    frame_num = 0
    ret, frame = video.read()
    #initialize video for writing purposes
    out = cv2.VideoWriter('ball_candidates_each_frame_from_beginning_yolo_kinematics_landing_RIGHT.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                          (frame.shape[1], frame.shape[0]))
    # undistorting image based on camera calibration
    frame = undistort(frame, False)
    # write image so detect.py can find it
    cv2.imwrite("current_frame.png", frame)

    for i in range(0, 60):
        ret, frame = video.read()
        frame_num +=1


    #loop through rest of frames for tracking
    flight_count = 0
    ball_info[frame_num] = {}
    while frame_num < 115:
        frame = undistort(frame, True)
        cv2.imwrite("current_frame.png", frame)
        print("frame num = ")
        print(frame_num)

        os.system("python3.7 /Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/detect.py"
                  " --weights '/Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/last_aug.pt'"
                  " --source /Users/ashley20/PycharmProjects/ThesisCameraCalibration/current_frame.png"
                  " --save-txt"
                  " --exist-ok")



        with open(
                '/Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/runs/detect/exp/labels/current_frame.txt',
                'r') as label_file:
            # adding identifier for each ball candidate in the frame
            label_num = 0

            #code to get number of detections after detect has been run
            with  open('/Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/runs/detect/exp/labels/current_frame_num_detections.txt',
                'r') as num_detections_file:
                for line in num_detections_file:
                    line=line.split()
                    num_detections = int(line[0])

            #if label file is empty, this means no ball candidates, so ball becomes predicted point
            if num_detections == 0:
                print("file is empty")
                print("num detections:", num_detections)
                x = ball_info[frame_num]['pred_pos_x']
                y = ball_info[frame_num]['pred_pos_y']
                ball_info[frame_num]['det_pos_x'] = x
                ball_info[frame_num]['det_pos_y'] = y
                prev_info = ball_info[frame_num - 1]
                vel_x = x - prev_info['det_pos_x']
                vel_y = y - prev_info['det_pos_y']
                ball_info[frame_num]['vel_x'] = vel_x
                ball_info[frame_num]['vel_y'] = vel_y
                ball_info[frame_num]['vel'] = math.sqrt(vel_x ** 2 + vel_y ** 2)
                ball_info[frame_num]['angle'] = math.atan(vel_y / vel_x) if vel_x != 0 else float("-inf")
                a_x = vel_x - prev_info['vel_x']
                a_y = vel_y - prev_info['vel_y']
                ball_info[frame_num]['accel_x'] = a_x
                ball_info[frame_num]['accel_y'] = a_y
                ball_info[frame_num]['accel'] = math.sqrt(a_x ** 2 + a_y ** 2)

                # now, finally have enough info to make prediction about next frame with all values
                ball_info[frame_num+1] = {}
                ball_info[frame_num + 1]['pred_pos_x'] = x + vel_x + 0.5 * a_x
                ball_info[frame_num + 1]['pred_pos_y'] = y + vel_y + 0.5 * a_y

                ball_info[frame_num]['is_pred'] = True
                ball_info[frame_num]['bb_width'] = None
                ball_info[frame_num]['bb_height'] = None

                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), 5)

            #else if there are ball candidates
            else:
                if frame_num == 60:
                    for line in label_file:
                        line = line.split()
                        x = float(line[1])
                        y = float(line[2])
                        dh, dw, _ = frame.shape
                        x = int(float(x) * dw)
                        y = int(float(y) * dh)
                        width = float(line[3]) * dw
                        height = float(line[4]) * dh
                        ball_info[frame_num]['det_pos_x'] = x
                        ball_info[frame_num]['det_pos_y'] = y
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), 5)
                        ball_info[frame_num]['is_pred'] = False
                        ball_info[frame_num]['bb_width'] = width
                        ball_info[frame_num]['bb_height'] = height

                        # if first frame, then assume velocity is negligible and position is same place
                        ball_info[frame_num + 1] = {}
                        ball_info[frame_num + 1]['pred_pos_x'] = x
                        ball_info[frame_num + 1]['pred_pos_y'] = y

                #first, create search window from predicted ball position
                elif frame_num > 60:
                    pred_ball_x = ball_info[frame_num]['pred_pos_x']
                    pred_ball_y = ball_info[frame_num]['pred_pos_y']

                    #code to find best candidate
                    all_dists = {}
                    for line in label_file:
                        line = line.split()
                        x = float(line[1])
                        y = float(line[2])
                        dh, dw, _ = frame.shape
                        width = float(line[3]) * dw
                        height = float(line[4]) * dh
                        x = int(float(x) * dw)
                        y = int(float(y) * dh)
                        print(x,y)
                        print(pred_ball_x, pred_ball_y)
                        dist = dist_formula((x,y), (pred_ball_x, pred_ball_y))
                        all_dists[(x,y)] = dist
                        print("dist:", dist)

                    #minimum key then becomes the ball candidate but only if distance is reasonably close to ball
                    min_key = min(all_dists, key=all_dists.get)
                    print("min key and distance")
                    print(min_key)
                    print(all_dists[min_key])



                    #if distance falls within threshold, take it as the ball; if not, then take the prediction
                    #adjusting threshold based on two criteria:
                    # 1. threshold based on avg. of width and height of boxes
                    # 2. threshold increases as string of predicted points gets bigger
                    if flight_count == 0:
                        threshold = 100
                    else:
                        #set initial threshold
                        threshold=100

                        fn = frame_num - 1
                        while ball_info[fn]['is_pred'] is True:
                            fn -= 1
                        #get previous bounding box width and height
                        curr_width = ball_info[fn]['bb_width']
                        curr_height = ball_info[fn]['bb_height']

                        #ideally, should always go in here
                        if curr_width is not None and curr_height is not None:
                            avg_dimension = (curr_height + curr_width) / 2
                            print("avg dimension: ", avg_dimension)
                            threshold = avg_dimension*5
                        print("threshold: ", threshold)

                        fn = frame_num
                        while ball_info[fn-1]['is_pred'] is True:
                            fn -= 1
                        time_steps = frame_num - fn
                        print('time_steps: ', time_steps)
                        #increasing threshold by 20% for each time point is predicted
                        for x in range(time_steps):
                            threshold*=1.1
                        #print("new threshold: ", threshold)
                        #center_thresh = (int(ball_info[frame_num-1]['det_pos_x']), int(ball_info[frame_num-1]['det_pos_y']))
                        # if frame_num == 69:
                        #     print(ball_info[frame_num-1])
                        #     print(center_thresh)
                        # cv2.circle(frame, center_thresh, int(threshold), (255,255,0), 10)
                        #if frame_num>=69:
                         #   cv2.imshow('f', frame)
                          #  cv2.waitKey(0)

                    if all_dists[min_key] < threshold:
                        x = int(min_key[0])
                        y = int(min_key[1])
                        ball_info[frame_num]['det_pos_x'] = x
                        ball_info[frame_num]['det_pos_y'] = y
                        ball_info[frame_num]['bb_width'] = width
                        ball_info[frame_num]['bb_height'] = height
                        ball_info[frame_num]['is_pred'] = False
                    else:
                        x = int(ball_info[frame_num]['pred_pos_x'])
                        y = int(ball_info[frame_num]['pred_pos_y'])
                        ball_info[frame_num]['det_pos_x'] = x
                        ball_info[frame_num]['det_pos_y'] = y
                        print("taking predicted point!")
                        ball_info[frame_num]['is_pred'] = True
                        #if predicted point, there is no known bounding box
                        ball_info[frame_num]['bb_width'] = None
                        ball_info[frame_num]['bb_height'] = None
                        #print(x)
                        #print(y)

                    print(x)
                    print(y)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), 5)
                    print("x", x)
                    print("y", y)

                    if flight_count == 0:
                        ball_info[frame_num]['is_pred'] = False

                        #if first frame, then assume velocity is negligible and position is same place
                        ball_info[frame_num + 1] = {}
                        ball_info[frame_num + 1]['pred_pos_x'] = x
                        ball_info[frame_num + 1]['pred_pos_y'] = y

                    if flight_count >= 1:
                        prev_info = ball_info[frame_num-1]

                        #reset velocity values if back to predicted, calculate them based on last detected values
                        if (prev_info['is_pred'] is True) and (ball_info[frame_num]['is_pred'] is False):
                            fn=frame_num-1
                            while ball_info[fn]['is_pred'] is True:
                                fn-=1
                            time_steps = frame_num-fn
                            print(fn)
                            print(time_steps)
                            vel_x = 2*(x-ball_info[fn]['det_pos_x'])/time_steps - ball_info[fn]['vel_x']
                            vel_y = 2*(y-ball_info[fn]['det_pos_y'])/time_steps - ball_info[fn]['vel_y']
                        else:
                            vel_x = x - prev_info['det_pos_x']
                            vel_y = y - prev_info['det_pos_y']
                        ball_info[frame_num]['vel_x'] = vel_x
                        ball_info[frame_num]['vel_y'] = vel_y
                        ball_info[frame_num]['vel'] = math.sqrt(vel_x**2 + vel_y**2)
                        ball_info[frame_num]['angle'] = math.atan(vel_y / vel_x) if vel_x != 0 else float("-inf")

                        # now, finally have enough info to make prediction about next frame, but assume accelration here is negligible
                        ball_info[frame_num + 1] = {}
                        ball_info[frame_num+1]['pred_pos_x'] = int(x + vel_x)
                        ball_info[frame_num+1]['pred_pos_y'] = int(y + vel_y)

                        #cv2.circle(frame, (int(x+vel_x), int(y+vel_y)), 5, (0, 0, 255), 5)


                    if flight_count >=2:
                        prev_info = ball_info[frame_num - 1]
                        # reset acceleration values if back to predicted, calculate them based on last detected values
                        if (prev_info['is_pred'] is True) and (ball_info[frame_num]['is_pred'] is False):
                            fn = frame_num - 1
                            while ball_info[fn]['is_pred'] is True:
                                fn -= 1
                            time_steps = frame_num - fn
                            print(fn)
                            print(time_steps)
                            a_x = (math.pow(vel_x,2)-math.pow(ball_info[fn]['vel_x'],2))/(2*(x-ball_info[fn]['det_pos_x']))
                            a_y = (math.pow(vel_y,2)-math.pow(ball_info[fn]['vel_y'],2))/(2*(y-ball_info[fn]['det_pos_y']))
                        else:
                            a_x = vel_x - prev_info['vel_x']
                            a_y = vel_y - prev_info['vel_y']
                        ball_info[frame_num]['accel_x'] = a_x
                        ball_info[frame_num]['accel_y'] = a_y
                        ball_info[frame_num]['accel'] = math.sqrt(a_x**2 + a_y**2)

                        # now, finally have enough info to make prediction about next frame with all values
                        ball_info[frame_num + 1]['pred_pos_x'] = x + vel_x + 0.5*a_x
                        ball_info[frame_num + 1]['pred_pos_y'] = y + vel_y + 0.5*a_y

                        #cv2.circle(frame, (int(x + vel_x + 0.5*a_x), int(y + vel_y + 0.5*a_y)), 5, (0, 0, 255), 5)

                    #print(ball_info[frame_num])
                    #if frame_num >= 60:
                     #   cv2.imshow('f', frame)
                      #  cv2.waitKey(0)

        for key in ball_info:
            if key < frame_num:
                x_pos = int(ball_info[key]['det_pos_x'])
                y_pos = int(ball_info[key]['det_pos_y'])
                #print(x_pos)
                #print(y_pos)
                cv2.circle(frame, (x_pos,y_pos), 5, (0,255,0), 5)

        #code to figure out landing
        if frame_num >= 63:
            curr_v_x = ball_info[frame_num]['vel_x']
            curr_v_y = ball_info[frame_num]['vel_y']
            prev_v_x = ball_info[frame_num-1]['vel_x']
            prev_v_y = ball_info[frame_num-1]['vel_y']
            prev_accel_x = ball_info[frame_num - 1]['accel_x']
            prev_accel_y = ball_info[frame_num-1]['accel_y']
            prev_v = ball_info[frame_num-1]['vel']
            prev_a = ball_info[frame_num-1]['accel']
            prev_x = ball_info[frame_num-1]['det_pos_x']
            prev_y = ball_info[frame_num-1]['det_pos_y']

            if curr_v_y < 0 and prev_v_y > 0:
                cv2.putText(frame, "LANDED!", (50,1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.circle(frame, (int(ball_info[frame_num]['det_pos_x']), int(ball_info[frame_num]['det_pos_y'])), 20, (0,0,255), -1)

                #now that landing frame has been found, try to interpolate exact landing pixel coordinate
                #first figure out time step to get from current velocity to 0 velocity
                time_step = (0-prev_v)/prev_a
                print(time_step)
                time_step=abs(time_step)

                #then apply that time step to position equations to get x and y coordinate prediction
                pred_landing_x = prev_x + prev_v_x*time_step + 0.5*prev_accel_x*math.pow(time_step, 2)
                pred_landing_y = prev_y + prev_v_y*time_step + 0.5 * prev_accel_y * math.pow(time_step, 2)

                print("landing point of prev frame:", (prev_x, prev_y))
                print("landing point predicted:", (pred_landing_x, pred_landing_y))
                cv2.putText(frame, "Predited Landing Point: (" + str(pred_landing_x) + ', ' + str(pred_landing_y) + ')', (50, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        #if frame_num >= 110:
            #print(ball_info)
            #cv2.imshow('f', frame)
            #cv2.waitKey(0)
        out.write(frame)
        ret, frame = video.read()
        flight_count+=1
        frame_num+=1
    out.release()

if __name__ == '__main__':
   #print("in ball tracking yolo")
   video_left = cv2.VideoCapture('Thesis_Data_Videos_Left/throwfar_2_292_behind_shot_on_160_left.MP4')
   video_right = cv2.VideoCapture('Thesis_Data_Videos_Right/throwfar_2_292_behind_shot_on_160_right.MP4')
   if not video_left.isOpened() or not video_right.isOpened():
       print("no video opened")
       exit()
   else: track_ball_kinematics(video_right)

