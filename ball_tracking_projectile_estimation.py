import numpy as np
import cv2
import math
import operator
from camera_calibration import undistort

#tracking for left frame
def tracking_left():

    video = cv2.VideoCapture('Thesis_Data_Videos_Left/throwfar_2_292_behind_shot_on_160_left.MP4')

    if not video.isOpened():
        print("no video opened")
        exit()

    frame_num = 0
    ret, old_frame = video.read()
    # undistorting image based on camera calibration
    old_frame = undistort(old_frame, True)

    #initial background subtractor, this prevents more noise than just normal subtraction
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=250, detectShadows=True)

    #looping through 50 frames to get frame when ball is leaving hand
    for i in range(0, 52):
        ret, old_frame = video.read()
        # undistorting image based on camera calibration
        old_frame = undistort(old_frame, True)
        backSub.apply(old_frame)
        frame_num += 1

    cv2.imshow("old frame", old_frame)
    cv2.waitKey(0)

    #back subtractor applied
    mask = backSub.apply(old_frame)
    #cv2.imshow("mask before thresholding", mask)
    #cv2.waitKey(0)

    #messed around with threshold to find one that makes ball most clear but also minimizes noise
    #increased window for blur
    blur = cv2.GaussianBlur(mask, (13, 13), 0)
    ret, binarized_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("binarized_image", binarized_image)
    cv2.waitKey(0)

    #4 = looks for pixels up, down, left, right, #8 looks for those pixels plus diagonal neighbors
    #not seeing much difference between the two so chose simpler one
    connectivity = 4

    # run connected components algorithm on binarized image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image, connectivity, cv2.CV_32S)
    #print("num labels", num_labels)
    #print("labels", labels.shape)
    #stats = matrix with length = # of labels, width = # of stats = 5
    #stats are leftmost x coordinate of bounding box, topmost y coordinate of bounding box,
    #horizontal size of bounding box, vertical size of bounding box, total area in pixels of component
    #access vis stats[label, COLUMN]
    #print("stats", stats.shape)
    #centroids = matrix with (x,y) of centroid of connected component
    #print("centroids", centroids.shape)

    #drawing all values collected by stats in RED
    for box in stats:
        x = box[cv2.CC_STAT_LEFT]
        y = box[cv2.CC_STAT_TOP]
        w = box[cv2.CC_STAT_WIDTH]
        h = box[cv2.CC_STAT_HEIGHT]
        cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("old frame with all stats", old_frame)
    cv2.waitKey(0)

    #append label number to each row of stats and centroids to preserve
    label_number_column = np.arange(num_labels).reshape((num_labels, 1))
    stats = np.append(stats, label_number_column, axis=1)
    centroids = np.append(centroids, label_number_column, axis=1)

    #extract rows from stats that fall within certain area - currently between 200 and 400
    min_area = 200
    max_area = 600
    stats = stats[np.where(stats[:,cv2.CC_STAT_AREA] > min_area)]
    stats = stats[np.where(stats[:, cv2.CC_STAT_AREA] < max_area)]
    #print("stats with large area", stats)
    #print(stats.shape)

    #extract rows from stats that fall within a certain ratio because since shot is circular, we are looking for bounding box where height and width are roughly the same
    min_ratio = 0.6
    max_ratio = 3
    stats = stats[np.where((stats[:, cv2.CC_STAT_HEIGHT]/stats[:, cv2.CC_STAT_WIDTH]) > min_ratio)]
    stats = stats[np.where((stats[:, cv2.CC_STAT_HEIGHT] / stats[:, cv2.CC_STAT_WIDTH]) < max_ratio)]
    #print("stats with updated ratios", stats)
    #print(stats.shape)

    #nested dictionary for each frame (key) and dict of info (obj)
    #info per frame = x_pos, y_pos, is_estimate
    ball_candidates = {}
    #nested dictionary for each frame (key) and dict of info (obj)
    #info per frame = v_x, v_y, v, alpha, accel_y
    trajectory_info = {}

    #tuple where first is value, second is number of values avg is calculated with
    avg_velocity_x = (0, 0)

    #draw centroids and bounding boxes on image at beginning
    for box in stats:
        curr_label = box[5]
        curr_centroid = (int(centroids[curr_label][0]), int(centroids[curr_label][1]))
        #print("curr centroid", curr_centroid)
        x = box[cv2.CC_STAT_LEFT]
        y = box[cv2.CC_STAT_TOP]
        w = box[cv2.CC_STAT_WIDTH]
        h = box[cv2.CC_STAT_HEIGHT]
        num_pixels = w*h
        #cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #cv2.circle(old_frame, curr_centroid, 20, (255, 0, 0), 2)
        #code to calculate percent fill of bounding box
        num_white_pixels = 0
        for col in range(x, x+w):
            for row in range(y, y+h):
                if binarized_image[row][col] == 255: num_white_pixels +=1
        #print("num white", num_white_pixels)
        #print("num pixels", num_pixels)
        #print("num white pixels", num_white_pixels/num_pixels)
        if (num_white_pixels/num_pixels) > 0.6:
            print("white ratio", num_white_pixels/num_pixels)
            ball_candidates[frame_num] = {
                "x_pos": curr_centroid[0],
                "y_pos": curr_centroid[1],
                "is_estimate": False
            }
            cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(old_frame, curr_centroid, 20, (255, 0, 0), 2)
            trajectory_info[frame_num] = {
                "v_x": float("-inf"),
                "v_y": float("-inf"),
                "v": float("-inf"),
                "alpha": float("inf"),
                "accel_y": float("-inf")
            }

    #print("initial ball candidates", ball_candidates[frame_num])
    #hard coding initial ball position and trajectory info for testing purposes
    # ball_candidates[frame_num] = {
    #     "x_pos": 1185,
    #     "y_pos": 677,
    #     "is_estimate": False
    # }
    # trajectory_info[frame_num] = {
    #     "v_x": float("-inf"),
    #     "v_y": float("-inf"),
    #     "v": float("-inf"),
    #     "alpha": float("inf"),
    #     "accel_y": float("-inf")
    # }
    cv2.imshow("old frame with centroids", old_frame)
    cv2.waitKey(0)

    #decrease blur as loop goes on because ball is getting smaller
    blur_window = [13,13]

    loop_num = 0
    while(1):
        #initializing important values
        loop_num+=1
        frame_num += 1

        prev_frame = frame_num - 1
        prev_prev_frame = frame_num - 2

        ret,frame = video.read()
        #undistorting image based on camera calibration
        frame = undistort(frame, isLeft=True)
        curr_backSub = backSub.apply(frame)
        #cv2.imshow("mask before thresholding", curr_backSub)
        #cv2.waitKey(0)

        #gradually reducing blur as ball gets smaller but making sure it stays a positive odd value
        blur_window = [int(x) for x in (list(map((lambda x: x*0.9), blur_window)))]
        if any(x%2==0 for x in blur_window) : blur_window = list(map((lambda x: x+1), blur_window))
        print("blur_window", blur_window)
        blur = cv2.GaussianBlur(curr_backSub, (blur_window[0], blur_window[1]), 0)
        #cv2.imshow("blur", blur)

        # get coordinates of ball in prev frame
        prev_ball_x = ball_candidates[prev_frame]['x_pos']
        prev_ball_y = ball_candidates[prev_frame]['y_pos']

        #determine search window for thresholding based on where ball is in prev frame, didnt work - caught too much noise
        # x = int(prev_ball_x - 25)
        # y = int(prev_ball_y - 25)
        # ball_roi = blur[y:y+50, x:x+50]
        # cv2.imshow("ball_roi", ball_roi)
        # print(ball_roi)
        # ret, img = cv2.threshold(ball_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # print("threshold value of roi", ret)
        #ret_updated, binarized_image = cv2.threshold(blur, ret, 255, cv2.THRESH_BINARY)

        #Otsu thresholding was best choice
        ret, binarized_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #print("threshold value of normal thresholding", ret_updated)
        # draw rectangle around threshold window = ORANGE
        #cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 165, 255), 5)

        cv2.imshow("binarized_image", binarized_image)
        cv2.waitKey(0)

        #retrieving necessary info from previous frame
        prev_velocity_x = trajectory_info[prev_frame]['v_x']
        prev_velocity_y = trajectory_info[prev_frame]['v_y']
        prev_accel_y = trajectory_info[prev_frame]['accel_y']

        #if first or second frame or if some error, will just assume estimate is prev location
        if loop_num == 1:
            ball_pos_estimate = (prev_ball_x, prev_ball_y)
        # estimate of centroid of ball in this frame if prev velocities exist, but no accel exists yet --> 1st order approx
        elif loop_num == 2:
            ball_pos_estimate = (prev_ball_x+prev_velocity_x, prev_ball_y+prev_velocity_y)
        #estimate of centroid of ball from one prev point --> 2nd order approx
        #elif loop_num == 3:
        else:
            ball_pos_estimate = (prev_ball_x + prev_velocity_x, prev_ball_y + prev_velocity_y - 0.5*prev_accel_y)
        #estimate of centroid of ball from two prev points --> 2nd order approx
        # else:
        #     prev_prev_ball_x = ball_candidates[prev_prev_frame]['x_pos']
        #     prev_prev_ball_y = ball_candidates[prev_prev_frame]['y_pos']
        #     prev_prev_vel_x = trajectory_info[prev_prev_frame]['v_x']
        #     prev_prev_vel_y = trajectory_info[prev_prev_frame]['v_y']
        #     prev_prev_accel_y = trajectory_info[prev_prev_frame]['accel_y']
        #     ball_pos_estimate_x = prev_prev_ball_x + prev_prev_vel_x + prev_velocity_x
        #     ball_pos_estimate_y = prev_prev_ball_y + prev_prev_vel_y + (0.5 * prev_prev_accel_y) + prev_velocity_y + (0.5 * prev_accel_y)
        #     ball_pos_estimate = (ball_pos_estimate_x, ball_pos_estimate_y)

        #print("ball pos estimate", ball_pos_estimate)
        # Projectile estimate = lime green circle
        #cv2.circle(frame, (int(ball_pos_estimate[0]), int(ball_pos_estimate[1])), 10, (20, 255, 57), 2)

        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image, connectivity, cv2.CV_32S)

        #all connected components found before trimming = pink
        # for box in stats:
        #     x = box[cv2.CC_STAT_LEFT]
        #     y = box[cv2.CC_STAT_TOP]
        #     w = box[cv2.CC_STAT_WIDTH]
        #     h = box[cv2.CC_STAT_HEIGHT]
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 255), 3)

        # append label number to each row of stats and centroids to preserve
        label_number_column = np.arange(num_labels).reshape((num_labels, 1))
        stats = np.append(stats, label_number_column, axis=1)
        centroids = np.append(centroids, label_number_column, axis=1)

        # extract rows from stats and centroids that fall within certain area - currently between 200 and 400
        #incrementally decrease area parameters
        min_area *=0.7
        max_area *=0.98
        #print("min_area", min_area)
        #print("max_area", max_area)
        stats = stats[np.where(stats[:, cv2.CC_STAT_AREA] > min_area)]
        stats = stats[np.where(stats[:, cv2.CC_STAT_AREA] < max_area)]
        #print("stats with large area", stats)
        #print(stats.shape)


        min_ratio *= 0.9
        max_ratio *= 1.1
        stats = stats[np.where((stats[:, cv2.CC_STAT_HEIGHT] / stats[:, cv2.CC_STAT_WIDTH]) > 0.6)]
        stats = stats[np.where((stats[:, cv2.CC_STAT_HEIGHT] / stats[:, cv2.CC_STAT_WIDTH]) < 3)]
        #print("stats with updated ratios", stats)
        #print(stats.shape)

        #code to trim based on ratio of white pixels but doesnt work well
        # stats_updated = []
        # for box in stats:
        #     curr_label = box[5]
        #     curr_centroid = (int(centroids[curr_label][0]), int(centroids[curr_label][1]))
        #     # print("curr centroid", curr_centroid)
        #     x = box[cv2.CC_STAT_LEFT]
        #     y = box[cv2.CC_STAT_TOP]
        #     w = box[cv2.CC_STAT_WIDTH]
        #     h = box[cv2.CC_STAT_HEIGHT]
        #     num_pixels = w * h
        #     # cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #     # cv2.circle(old_frame, curr_centroid, 20, (255, 0, 0), 2)
        #     # code to calculate percent fill of bounding box
        #     num_white_pixels = 0
        #     for col in range(x, x + w):
        #         for row in range(y, y + h):
        #             if binarized_image[row][col] == 255: num_white_pixels += 1
        #     # print("num white", num_white_pixels)
        #     # print("num pixels", num_pixels)
        #     # print("num white pixels", num_white_pixels/num_pixels)
        #     if (num_white_pixels / num_pixels) > 0.5:
        #         stats_updated.append(box)
        #print("stats after pixel trim", stats)

        #draws all components found after trimming = RED
        # for box in stats:
        #     x = box[cv2.CC_STAT_LEFT]
        #     y = box[cv2.CC_STAT_TOP]
        #     w = box[cv2.CC_STAT_WIDTH]
        #     h = box[cv2.CC_STAT_HEIGHT]
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # get centroid of ball from previous frame
        prev_ball_x = ball_candidates[frame_num - 1]['x_pos']
        prev_ball_y = ball_candidates[frame_num - 1]['y_pos']

        # draw centroids and bounding boxes on image
        # boolean to see if estimate needs to be used
        #found_ball = False
        centroid_dists_prev = {}
        for box in stats:
            curr_label = box[5]
            curr_centroid = (int(centroids[curr_label][0]), int(centroids[curr_label][1]))
            #Method 1 = calculate dist from estimate to current centroid its looking at
            #centroids_dist_projectile = dist_formula(ball_pos_estimate, curr_centroid)
            #centroid_dists_prev[curr_label] = centroids_dist_projectile
            #print("centroids dist", centroids_dist_projectile)

            #Method 2 = better method = to calculate dist from prev point, we are adding to list so we can eventually find the min
            prev_centroid = (prev_ball_x, prev_ball_y)
            centroids_dist_prev_centroid = dist_formula(prev_centroid, curr_centroid)
            centroid_dists_prev[curr_label] = centroids_dist_prev_centroid


        #get minimum centroid distance from prev point out of the candidates for the ball
        #print(min(centroid_dists_prev.items(), key=operator.itemgetter(1)))
        best_centroid_label = min(centroid_dists_prev.items(), key=operator.itemgetter(1))[0]
        best_centroid_stat = stats[np.where(stats[:, 5] == best_centroid_label)]
        #print("best_centroid_stat", best_centroid_stat)

        # draws best centroid = LIGHT BLUE
        # x = best_centroid_stat[0][cv2.CC_STAT_LEFT]
        # y = best_centroid_stat[0][cv2.CC_STAT_TOP]
        # w = best_centroid_stat[0][cv2.CC_STAT_WIDTH]
        # h = best_centroid_stat[0][cv2.CC_STAT_HEIGHT]
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 5)

        #look at closest centroid that matches the criteria
        #if too far away from previous point, then take projectile estimate instead
        best_centroid_dist = min(centroid_dists_prev.items(), key=operator.itemgetter(1))[1]
        #print("best centroid dist", best_centroid_dist)
        best_centroid = (centroids[best_centroid_label][0], centroids[best_centroid_label][1])
        #print("best centroid", best_centroid)
        #print("centroid dist radius", trajectory_info[frame_num-1]['v'])

        #tried to relate the window to the prev velocity but too buggy, just chose a value instead
        #best_centroid_dist_estimate = 70 if trajectory_info[frame_num-1]['v'] == float('-inf') else trajectory_info[frame_num-1]['v']

        best_centroid_dist_estimate = 70
        #testing if best distance radius should be related to velocity of object
        if best_centroid_dist < best_centroid_dist_estimate:
            found_ball = True
            # print("centroids dist", centroids_dist)
            # print("curr centroid", curr_centroid)
            # cv2.circle(frame, curr_centroid, 20, (255, 0, 0), 2)
            # x = best_centroid_stat[0][cv2.CC_STAT_LEFT]
            # y = best_centroid_stat[0][cv2.CC_STAT_TOP]
            # w = best_centroid_stat[0][cv2.CC_STAT_WIDTH]
            # h = best_centroid_stat[0][cv2.CC_STAT_HEIGHT]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # ball_cand#idates[frame_num].append(curr_centroid)
            ball_candidates[frame_num] = {
                'x_pos': best_centroid[0],
                'y_pos': best_centroid[1],
                'is_estimate': False
            }
            #circle best centroid to verify = DARK BLUE
            #cv2.circle(frame, (int(best_centroid[0]), int(best_centroid[1])), 20, (139, 0, 0), 3)

            # print("ball_candidates at the moment", ball_candidates[frame_num])
            #print("found ball", found_ball)

        #best centroid found probably is not the ball, take estimate instead
        else:
            found_ball = False
            # print("centroids dist", centroids_dist)
            # print("curr centroid", curr_centroid)
            # cv2.circle(frame, curr_centroid, 20, (255, 0, 0), 2)
            #x = box[cv2.CC_STAT_LEFT]
            #y = box[cv2.CC_STAT_TOP]
            #w = box[cv2.CC_STAT_WIDTH]
            #h = box[cv2.CC_STAT_HEIGHT]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # ball_cand#idates[frame_num].append(curr_centroid)
            print("we are taking the estimate")
            ball_candidates[frame_num] = {
                'x_pos': ball_pos_estimate[0],
                'y_pos': ball_pos_estimate[1],
                'is_estimate': True
            }


        #calculate current v_x and v_y
        v_x = ball_candidates[frame_num]['x_pos'] - ball_candidates[frame_num-1]['x_pos']
        #can only calculate updated velocity once acceleration is calculated
        v_y = ball_candidates[frame_num]['y_pos'] - ball_candidates[frame_num-1]['y_pos']

        # calculate initial velocity based on assumption it was launched in prev frame - this only happens in second loop
        if loop_num == 2:
            #print("curr frame x pos", ball_candidates[frame_num]['x_pos'])
            #print("prev frame x pos", ball_candidates[frame_num-1]['x_pos'])
            initial_vel_x = v_x
            #units here are not correct
            initial_vel_y = v_y + 9.8*60
            launch_angle = math.atan(initial_vel_y/initial_vel_x)
            launch_angle_deg = math.degrees(launch_angle)
            initial_vel = initial_vel_y/initial_vel_x
            #print("initial vel x", initial_vel_x)
            #print("initial vel y", initial_vel_y)
            #print("launch angle rad", launch_angle, "deg ", launch_angle_deg)
            #print("initial vel", initial_vel)

        if loop_num == 3:
            initial_accel = v_y - trajectory_info[frame_num-1]['v_y']
            print("intial accel in frames/sec^2", initial_accel)


        # code to predict pixels to meters conversion based on acceleration
        # if loop_num >= 3:
        #     #get current and prev velocity and convert to pixels/sec
        #     v_y_curr_pps = v_y*60
        #     v_y_prev_pps = trajectory_info[frame_num-1]['v_y']*60
        #     #unit = pixels/sec^2
        #     accel_pred_ppss = v_y_curr_pps - v_y_prev_pps
        #     if accel_pred_ppss != 0:
        #         print("accel pred pps", accel_pred_ppss)
        #         #meters/pixel = (m/sec^2) / (pixels/sec^2)
        #         meters_per_pixel = abs(9.8/accel_pred_ppss)
        #         print("meters per pixel", meters_per_pixel)

        # update average x velocity if ball is from bounding box so we are not just calculating velocity based on estimates
        if found_ball:
            new_num_values = avg_velocity_x[1] + 1
            new_avg = (avg_velocity_x[0] + v_x) / new_num_values
            avg_velocity_x = (new_avg, new_num_values)
            #print("avg velocity x", avg_velocity_x)

        # update trajectory info
        trajectory_info[frame_num] = {
            "v_x": v_x,
            "v_y": v_y,
            "v": math.sqrt(v_x**2 + v_y**2),
            "alpha": math.atan(v_y/v_x) if v_x != 0 else float("-inf"),
            "accel_y": v_y - trajectory_info[frame_num-1]['v_y']
        }

        #print("trajectory info", trajectory_info[frame_num])
        #print("ball candidates", ball_candidates[frame_num])

        #circles the chosen location for the ball = Yellow
        cv2.circle(frame, (int(ball_candidates[frame_num]['x_pos']), int(ball_candidates[frame_num]['y_pos'])), 10, (0, 255, 255), 4)


        current_vel_x = "X Velocity: " + str(trajectory_info[frame_num]['v_x']) + " pixels/frame"
        current_vel_y = "Y Velocity: " + str(trajectory_info[frame_num]['v_y']) + " pixels/frame"
        current_vel = "Velocity: " + str(trajectory_info[frame_num]['v']) + " pixels/frame"
        current_angle = "Angle: " + str(trajectory_info[frame_num]['alpha']) + " radians "
        current_accel = "Acceleration: " + str(trajectory_info[frame_num]['accel_y']) + " pixels/frame^2"
        is_estimate_string = "Best Ball Point = Projectile Estimate" if ball_candidates[frame_num]['is_estimate'] else "Best Ball Point = Best Component based on Prev Point"
        curr_coords = "Ball Coordinates: " + str(ball_candidates[frame_num]['x_pos']) + ", " + str(ball_candidates[frame_num]['y_pos'])
        curr_coords_color = (20, 255, 57) if ball_candidates[frame_num]['is_estimate'] else (255,255,0)
        cv2.putText(frame, current_vel_x, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_4)
        cv2.putText(frame, current_vel_y, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_4)
        cv2.putText(frame, current_vel, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_4)
        cv2.putText(frame, current_angle, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_4)
        cv2.putText(frame, current_accel, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_4)
        cv2.putText(frame, is_estimate_string, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, curr_coords_color, 3, cv2.LINE_4)
        cv2.putText(frame, curr_coords, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_4)

        cv2.imshow("current frame with data", frame)
        cv2.waitKey(0)




#tracking for right frame
def tracking_right():
    # params for ShiTomasi corner detection
    # feature_params = dict( maxCorners = 100,
    #                        qualityLevel = 0.3,
    #                        minDistance = 7,
    #                        blockSize = 7 )
    # # Parameters for lucas kanade optical flow
    # lk_params = dict( winSize  = (15,15),
    #                   maxLevel = 2,
    #                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # # Create some random colors
    # color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    video = cv2.VideoCapture('Thesis_Data_Videos_Right/throwfar_2_292_behind_shot_on_160_right.MP4')

    if not video.isOpened():
        print("no video opened")
        exit()

    frame_num = 0
    ret, old_frame = video.read()
    # undistorting image based on camera calibration
    old_frame = undistort(old_frame, False)


    #initial background subtractor, this prevents more noise than just normal subtraction
    #utilizing background subtraction here takes advantage of temporal nature of these images
    #how in a small window of time, the ball moves much more than anything else
    #PROBLEM = when ball blends in with its background
    backSub = cv2.createBackgroundSubtractorMOG2(history = 1000, varThreshold=500, detectShadows=True)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #this function is using Shi Tomasi corner detector, but we want circles to track
    #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #print("shi tomasi p0")
    #print(p0.shape)
    #looping through 50 frames to get point when ball is leaving hand
    for i in range(0, 59):
        ret, old_frame = video.read()
        # undistorting image based on camera calibration
        old_frame = undistort(old_frame, False)
        backSub.apply(old_frame)
        frame_num += 1

    cv2.imshow("old frame", old_frame)
    cv2.waitKey(0)

    #back subtractor applied
    mask = backSub.apply(old_frame)
    #cv2.imshow("mask before thresholding", mask)
    #cv2.waitKey(0)

    #messed around with threshold to find one that makes ball most clear but also
    #minimizes noise
    #increased window for blur
    blur = cv2.GaussianBlur(mask, (13, 13), 0)
    #ret becomes ideal threshold of window of ball estimate
    ret, binarized_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("binarized_image", binarized_image)
    cv2.waitKey(0)

    #p0 = find.find_shot(old_frame, mask)
    #print("shape", len(mask.shape))

    #run connected components algorithm on binarized image

    #choose between 4 and 8 for connectivity
    #4 = looks for pixels up, down, left, right, #8 looks for those pixels plus diagonal neighbors
    connectivity = 4

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image, connectivity, cv2.CV_32S)
    #print("num labels", num_labels)
    #print("labels", labels.shape)
    #stats = matrix with length = # of labels, width = # of stats = 5
    #stats are leftmost x coordinate of bounding box, topmost y coordinate of bounding box,
    #horizontal size of bounding box, vertical size of bounding box, total area in pixels of component
    #access vis stats[label, COLUMN]
    #print("stats", stats.shape)
    #centroids = matrix with (x,y) of centroid of connected component
    #print("centroids", centroids.shape)
    for box in stats:
        x = box[cv2.CC_STAT_LEFT]
        y = box[cv2.CC_STAT_TOP]
        w = box[cv2.CC_STAT_WIDTH]
        h = box[cv2.CC_STAT_HEIGHT]
        cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("old frame with all stats", old_frame)
    cv2.waitKey(0)

    #append label number to each row of stats and centroids to preserve
    label_number_column = np.arange(num_labels).reshape((num_labels, 1))
    stats = np.append(stats, label_number_column, axis=1)
    centroids = np.append(centroids, label_number_column, axis=1)

    #copy of stats
    stats_orig = np.copy(stats)

    #extract rows from stats and centroids that fall within certain area - currently between 200 and 400
    min_area = 100
    max_area = 300
    stats = stats[np.where(stats[:,cv2.CC_STAT_AREA] > min_area)]
    stats = stats[np.where(stats[:, cv2.CC_STAT_AREA] < max_area)]
    #print("stats with large area", stats)
    #print(stats.shape)

    min_ratio = 0.6
    max_ratio = 3
    #stats = stats[np.where((stats[:, cv2.CC_STAT_HEIGHT]/stats[:, cv2.CC_STAT_WIDTH]) > min_ratio)]
    #stats = stats[np.where((stats[:, cv2.CC_STAT_HEIGHT] / stats[:, cv2.CC_STAT_WIDTH]) < max_ratio)]
    #print("stats with updated ratios", stats)
    #print(stats.shape)

    #dictionary for each frame (key) and its point(s) for ball (obj - list)
    #ball_candidates = defaultdict(list)
    ball_candidates = {}
    #nested dictionary for each frame (key) and dict of info (obj)
    #info in frame = v_x, v_y, v, alpha
    trajectory_info = {}

    #tuple where first is value, second is number of values avg is calculated with
    avg_velocity_x = (0, 0)


    #draw centroids and bounding boxes on image at beginning
    for box in stats:
        curr_label = box[5]
        curr_centroid = (int(centroids[curr_label][0]), int(centroids[curr_label][1]))
        #print("curr centroid", curr_centroid)
        x = box[cv2.CC_STAT_LEFT]
        y = box[cv2.CC_STAT_TOP]
        w = box[cv2.CC_STAT_WIDTH]
        h = box[cv2.CC_STAT_HEIGHT]
        num_pixels = w*h
        cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #cv2.circle(old_frame, curr_centroid, 20, (255, 0, 0), 2)
        #code to calculate percent fill of bounding box
        num_white_pixels = 0
        for col in range(x, x+w):
            for row in range(y, y+h):
                if binarized_image[row][col] == 255: num_white_pixels +=1
        #print("num white", num_white_pixels)
        #print("num pixels", num_pixels)
        #print("num white pixels", num_white_pixels/num_pixels)
        if (num_white_pixels/num_pixels) > 0.6:
            print("white ratio", num_white_pixels/num_pixels)
            #ball_candidates[frame_num].append(curr_centroid)
            print(curr_centroid)
            if curr_centroid[0] > 600:
                ball_candidates[frame_num] = {
                    "x_pos": curr_centroid[0],
                    "y_pos": curr_centroid[1],
                    "is_estimate": False
                }
                cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(old_frame, curr_centroid, 20, (255, 0, 0), 2)
                trajectory_info[frame_num] = {
                    "v_x": float("-inf"),
                    "v_y": float("-inf"),
                    "v": float("-inf"),
                    "alpha": float("inf"),
                    "accel_y": float("-inf")
                }

    cv2.imshow("old frame with centroids", old_frame)
    cv2.waitKey(0)

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #print("p0:")
    #print(p0)
    #print("-------")
    # Create a mask image for drawing purposes
    #mask = np.zeros_like(old_frame)

    #setting points to type float32 to match what shi tomasi returns
    # p0 = np.array([[[1235, 725]],
    #                 [[1227, 728]],
    #                 [[1224, 729]],
    #                 [[1228, 734]],
    #                 [[1233, 742]],
    #                 [[1242, 733]],
    #                 [[1242, 729]],
    #                 [[1238, 724]],
    #                 [[1234, 734]],
    #                 [[1237, 724]],
    #                 [[1231, 729]],
    #                 [[1242, 728]]
    #                 ], dtype='f')
    #print("updated p0:")
    #print(p0.shape)

    #decrease blur as loop goes on because ball is getting smaller
    blur_window = [5,5]

    loop_num = 0
    while(1):
        loop_num+=1
        ret,frame = video.read()
        #undistorting image based on camera calibration
        frame = undistort(frame, False)
        frame_num += 1
        curr_backSub = backSub.apply(frame)
        #cv2.imshow("mask before thresholding", mask)
        #cv2.waitKey(0)
        #gradually reducing blur as ball gets smaller but making sure it stays a positive odd value
        blur_window = [int(x) for x in (list(map((lambda x: x*0.9), blur_window)))]
        if any(x%2==0 for x in blur_window) : blur_window = list(map((lambda x: x+1), blur_window))
        print("blur_window", blur_window)
        blur = cv2.GaussianBlur(curr_backSub, (blur_window[0], blur_window[1]), 0)
        ret, binarized_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("binarized_image", binarized_image)
        cv2.waitKey(0)

        #predict where ball should be
        prev_frame = frame_num - 1
        #coordinates of ball in prev frame
        prev_ball_x = ball_candidates[prev_frame]['x_pos']
        prev_ball_y = ball_candidates[prev_frame]['y_pos']
        #velocity
        prev_velocity_x = trajectory_info[prev_frame]['v_x']
        prev_velocity_y = trajectory_info[prev_frame]['v_y']

        prev_accel_y = trajectory_info[prev_frame]['accel_y']


        #if first or second frame or if some error, will just assume estimate is prev location
        #if math.isinf(trajectory_info[prev_frame]['v_x']):
        if loop_num == 1:
            ball_pos_estimate = (prev_ball_x, prev_ball_y)
        # estimate of centroid of ball in this frame if prev velocities exist, but no accel exists yet
        elif loop_num == 2:
            ball_pos_estimate = (prev_ball_x+prev_velocity_x, prev_ball_y+prev_velocity_y)
        else:
            ball_pos_estimate = (prev_ball_x + prev_velocity_x, prev_ball_y + prev_velocity_y + 0.5*prev_accel_y)
        print("ball pos estimate", ball_pos_estimate)
        #Projectile estimate = lime green circle
        cv2.circle(frame, (int(ball_pos_estimate[0]), int(ball_pos_estimate[1])), 10, (20, 255, 57), 5)

        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image, connectivity, cv2.CV_32S)

        #all connected components found before trimming = pink
        # for box in stats:
        #     x = box[cv2.CC_STAT_LEFT]
        #     y = box[cv2.CC_STAT_TOP]
        #     w = box[cv2.CC_STAT_WIDTH]
        #     h = box[cv2.CC_STAT_HEIGHT]
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 255), 3)

        # append label number to each row of stats and centroids to preserve
        label_number_column = np.arange(num_labels).reshape((num_labels, 1))
        stats = np.append(stats, label_number_column, axis=1)
        centroids = np.append(centroids, label_number_column, axis=1)

        # extract rows from stats and centroids that fall within certain area - currently between 200 and 400
        #incrementally decrease area parameters
        min_area *=0.7
        max_area *=0.98
        #print("min_area", min_area)
        #print("max_area", max_area)
        stats = stats[np.where(stats[:, cv2.CC_STAT_AREA] > min_area)]
        stats = stats[np.where(stats[:, cv2.CC_STAT_AREA] < max_area)]
        #print("stats with large area", stats)
        #print(stats.shape)


        min_ratio *= 0.9
        max_ratio *= 1.1
        stats = stats[np.where((stats[:, cv2.CC_STAT_HEIGHT] / stats[:, cv2.CC_STAT_WIDTH]) > 0.6)]
        stats = stats[np.where((stats[:, cv2.CC_STAT_HEIGHT] / stats[:, cv2.CC_STAT_WIDTH]) < 3)]
        #print("stats with updated ratios", stats)
        #print(stats.shape)

        #code to trim based on ratio of white pixels but doesnt work well
        # stats_updated = []
        # for box in stats:
        #     curr_label = box[5]
        #     curr_centroid = (int(centroids[curr_label][0]), int(centroids[curr_label][1]))
        #     # print("curr centroid", curr_centroid)
        #     x = box[cv2.CC_STAT_LEFT]
        #     y = box[cv2.CC_STAT_TOP]
        #     w = box[cv2.CC_STAT_WIDTH]
        #     h = box[cv2.CC_STAT_HEIGHT]
        #     num_pixels = w * h
        #     # cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #     # cv2.circle(old_frame, curr_centroid, 20, (255, 0, 0), 2)
        #     # code to calculate percent fill of bounding box
        #     num_white_pixels = 0
        #     for col in range(x, x + w):
        #         for row in range(y, y + h):
        #             if binarized_image[row][col] == 255: num_white_pixels += 1
        #     # print("num white", num_white_pixels)
        #     # print("num pixels", num_pixels)
        #     # print("num white pixels", num_white_pixels/num_pixels)
        #     if (num_white_pixels / num_pixels) > 0.5:
        #         stats_updated.append(box)

        #stats = np.array(stats_updated)
        print("stats after pixel trim", stats)

        #draws all components found after trimming = RED
        # for box in stats:
        #     x = box[cv2.CC_STAT_LEFT]
        #     y = box[cv2.CC_STAT_TOP]
        #     w = box[cv2.CC_STAT_WIDTH]
        #     h = box[cv2.CC_STAT_HEIGHT]
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # only look at stats close to centroid from prev frame
        # get centroid of ball from previous frame
        prev_ball = ball_candidates[frame_num - 1]
        prev_ball_x = prev_ball['x_pos']
        prev_ball_y = prev_ball['y_pos']
        search_window_x = [prev_ball_x - 50, prev_ball_x + 50]
        search_window_y = [prev_ball_y - 50, prev_ball_y + 50]

        # draw centroids and bounding boxes on image
        # boolean to see if estimate needs to be used
        found_ball = False
        centroid_dists_prev = {}
        for box in stats:
            curr_label = box[5]
            curr_centroid = (int(centroids[curr_label][0]), int(centroids[curr_label][1]))
            #calculate dist from estimate to current centroid its looking at
            centroids_dist_projectile = dist_formula(ball_pos_estimate, curr_centroid)
            #print("centroids dist", centroids_dist_projectile)

            prev_centroid = (prev_ball_x, prev_ball_y)
            centroids_dist_prev_centroid = dist_formula(prev_centroid, curr_centroid)
            centroid_dists_prev[curr_label] = centroids_dist_prev_centroid

            # if centroids_dist_prev_centroid < 50:
            #     found_ball = True
            #     #print("centroids dist", centroids_dist)
            #     #print("curr centroid", curr_centroid)
            #     #cv2.circle(frame, curr_centroid, 20, (255, 0, 0), 2)
            #     x = box[cv2.CC_STAT_LEFT]
            #     y = box[cv2.CC_STAT_TOP]
            #     w = box[cv2.CC_STAT_WIDTH]
            #     h = box[cv2.CC_STAT_HEIGHT]
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            #     #ball_cand#idates[frame_num].append(curr_centroid)
            #     ball_candidates[frame_num] = {
            #         'x_pos': curr_centroid[0],
            #         'y_pos': curr_centroid[1],
            #         'is_estimate': False
            #     }
            #
            #     #print("ball_candidates at the moment", ball_candidates[frame_num])
            #     print("found ball", found_ball)
            #just drawing test to see potential candidates not related to distance
            # else:
            #     x = box[cv2.CC_STAT_LEFT]
            #     y = box[cv2.CC_STAT_TOP]
            #     w = box[cv2.CC_STAT_WIDTH]
            #     h = box[cv2.CC_STAT_HEIGHT]
            #     #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        #get label minimum centroid distance
        print(min(centroid_dists_prev.items(), key=operator.itemgetter(1)))
        best_centroid_label = min(centroid_dists_prev.items(), key=operator.itemgetter(1))[0]
        best_centroid_stat = stats[np.where(stats[:, 5] == best_centroid_label)]
        print("best_centroid_stat", best_centroid_stat)
        x = best_centroid_stat[0][cv2.CC_STAT_LEFT]
        y = best_centroid_stat[0][cv2.CC_STAT_TOP]
        w = best_centroid_stat[0][cv2.CC_STAT_WIDTH]
        h = best_centroid_stat[0][cv2.CC_STAT_HEIGHT]
        #draws best centroid = LIGHT BLUE
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 5)

        #look at closest centroid that matches the criteria
        #if too far away from previous point, then take prediction instead
        best_centroid_dist = min(centroid_dists_prev.items(), key=operator.itemgetter(1))[1]
        print("best centroid dist", best_centroid_dist)
        best_centroid = (centroids[best_centroid_label][0], centroids[best_centroid_label][1])
        print("best centroid", best_centroid)
        print("centroid dist radius", trajectory_info[frame_num-1]['v'])

        #best_centroid_dist_estimate = 70 if trajectory_info[frame_num-1]['v'] == float('-inf') else trajectory_info[frame_num-1]['v']
        best_centroid_dist_estimate = 70
        #testing if best distance radius should be related to velocity of object
        if best_centroid_dist < best_centroid_dist_estimate:
            found_ball = True
            # print("centroids dist", centroids_dist)
            # print("curr centroid", curr_centroid)
            # cv2.circle(frame, curr_centroid, 20, (255, 0, 0), 2)
            x = best_centroid_stat[0][cv2.CC_STAT_LEFT]
            y = best_centroid_stat[0][cv2.CC_STAT_TOP]
            w = best_centroid_stat[0][cv2.CC_STAT_WIDTH]
            h = best_centroid_stat[0][cv2.CC_STAT_HEIGHT]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # ball_cand#idates[frame_num].append(curr_centroid)
            ball_candidates[frame_num] = {
                'x_pos': best_centroid[0],
                'y_pos': best_centroid[1],
                'is_estimate': False
            }
            #circle best centroid to verify = DARK BLUE
            #cv2.circle(frame, (int(best_centroid[0]), int(best_centroid[1])), 20, (139, 0, 0), 3)

            # print("ball_candidates at the moment", ball_candidates[frame_num])
            print("found ball", found_ball)

        #best centroid found probably is not the ball, take estimate instead
        else:
            found_ball = False
            # print("centroids dist", centroids_dist)
            # print("curr centroid", curr_centroid)
            # cv2.circle(frame, curr_centroid, 20, (255, 0, 0), 2)
            #x = box[cv2.CC_STAT_LEFT]
            #y = box[cv2.CC_STAT_TOP]
            #w = box[cv2.CC_STAT_WIDTH]
            #h = box[cv2.CC_STAT_HEIGHT]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # ball_cand#idates[frame_num].append(curr_centroid)
            print("we are taking the estimate")
            ball_candidates[frame_num] = {
                'x_pos': ball_pos_estimate[0],
                'y_pos': ball_pos_estimate[1],
                'is_estimate': True
            }

        #if no good candidate is found, make it the estimate instead
        # if found_ball is False:
        #     print("found ball", found_ball)
        #     ball_candidates[frame_num] = {
        #         'x_pos': ball_pos_estimate[0],
        #         'y_pos': ball_pos_estimate[1],
        #         'is_estimate': True
        #     }
        # print("ball points dict", ball_candidates)


        #calculate current v_x and v_y
        v_x = ball_candidates[frame_num]['x_pos'] - ball_candidates[frame_num-1]['x_pos']
        #can only calculate updated velocity once acceleration is calculated
        v_y = ball_candidates[frame_num]['y_pos'] - ball_candidates[frame_num-1]['y_pos']

        initial_vel_x = float("-inf")
        initial_vel_y = float("-inf")
        initial_accel = float("-inf")

        # calculate initial velocity based on assumption it was launched in prev frame
        print("loop num", loop_num)
        # if loop_num == 2:
        #     print("curr frame x pos", ball_candidates[frame_num]['x_pos'])
        #     print("prev frame x pos", ball_candidates[frame_num-1]['x_pos'])
        #     initial_vel_x = v_x
        #     #units here are not correct
        #     initial_vel_y = v_y + 9.8*60
        #     launch_angle = math.atan(initial_vel_y/initial_vel_x)
        #     launch_angle_deg = math.degrees(launch_angle)
        #     initial_vel = initial_vel_y/initial_vel_x
        #     print("initial vel x", initial_vel_x)
        #     print("initial vel y", initial_vel_y)
        #     print("launch angle rad", launch_angle, "deg ", launch_angle_deg)
        #     print("initial vel", initial_vel)

        if loop_num == 3:
            initial_accel = v_y - trajectory_info[frame_num-1]['v_y']
            print("intial accel in frames/sec^2", initial_accel)


        # code to predict pixels to meters conversion based on acceleration
        if loop_num >= 3:
            #get current and prev velocity and convert to pixels/sec
            v_y_curr_pps = v_y*60
            v_y_prev_pps = trajectory_info[frame_num-1]['v_y']*60
            #unit = pixels/sec^2
            accel_pred_ppss = v_y_curr_pps - v_y_prev_pps
            if accel_pred_ppss != 0:
                print("accel pred pps", accel_pred_ppss)
                #meters/pixel = (m/sec^2) / (pixels/sec^2)
                meters_per_pixel = abs(9.8/accel_pred_ppss)
                print("meters per pixel", meters_per_pixel)

        # update average x velocity if ball is from bounding box
        if found_ball:
            new_num_values = avg_velocity_x[1] + 1
            new_avg = (avg_velocity_x[0] + v_x) / new_num_values
            avg_velocity_x = (new_avg, new_num_values)
            print("avg velocity x", avg_velocity_x)

        # update trajectory info
        trajectory_info[frame_num] = {
            "v_x": v_x,
            "v_y": v_y,
            "v": math.sqrt(v_x**2 + v_y**2),
            "alpha": math.atan(v_y/v_x) if v_x != 0 else float("-inf"),
            "accel_y": v_y - trajectory_info[frame_num-1]['v_y']
        }

        print("trajectory info", trajectory_info[frame_num])
        print("ball candidates", ball_candidates[frame_num])

        cv2.circle(frame, (int(ball_candidates[frame_num]['x_pos']), int(ball_candidates[frame_num]['y_pos'])), 10, (0, 255, 255), 2)


        current_vel_x = "X Velocity: " + str(trajectory_info[frame_num]['v_x']) + " pixels/frame"
        current_vel_y = "Y Velocity: " + str(trajectory_info[frame_num]['v_y']) + " pixels/frame"
        current_vel = "Velocity: " + str(trajectory_info[frame_num]['v']) + " pixels/frame"
        current_angle = "Angle: " + str(trajectory_info[frame_num]['alpha']) + " radians "
        current_accel = "Acceleration: " + str(trajectory_info[frame_num]['accel_y']) + " pixels/frame^2"
        is_estimate_string = "Best Ball Point = Projectile Estimate" if ball_candidates[frame_num]['is_estimate'] else "Best Ball Point = Best Component based on Prev Point"
        curr_coords = "Ball Coordinates: " + str(ball_candidates[frame_num]['x_pos']) + ", " + str(ball_candidates[frame_num]['y_pos'])
        curr_coords_green = (20, 255, 57)
        curr_coords_blue = (20, 255, 57)
        cv2.putText(frame, current_vel_x, (50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, current_vel_y, (50, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, current_vel, (50, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, current_angle, (50, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, current_accel, (50, 1200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, is_estimate_string, (50, 1250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame, curr_coords, (50, 1300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

        cv2.imshow("current frame with centroids", frame)
        cv2.waitKey(0)



        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # calculate optical flow
        # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # print("p1:", p1)
        # print(p1.shape)
        # print("st:", st)
        # # Select good points
        # if p1 is not None:
        #     good_new = p1[st==1]
        #     good_old = p0[st==1]
        # # draw the tracks
        # for i,(new,old) in enumerate(zip(good_new, good_old)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        #     frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
        # img = cv2.add(frame, mask)
        #
        # cv2.imshow('frame', img)
        # cv2.waitKey(0)
        # #k = cv2.waitKey(30) & 0xff
        # #if k == 27:
        # #  break
        # # Now update the previous frame and previous points
        # old_gray = frame_gray.copy()
        # p0 = good_new.reshape(-1,1,2)



#function to get pixel at point where image is clicked
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        #cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        #cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                   # 1.0, (0, 0, 0), thickness=1)
        #cv2.imshow("image", img)
        print(x, y)

#associated code to run mouse click function if needed later
#cv2.namedWindow("image")
#cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
#cv2.imshow("image", old_frame)
#cv2.waitKey(0)

#distance formula that takes 2 tuples and returns value
def dist_formula(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)