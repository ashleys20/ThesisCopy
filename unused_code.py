

#RANSAC, path prediction, etc.
# def track_ball(video):
#
#     # ransac_ball_candidates = [[0.641927, 0.50625], [0.635417, 0.488194], [0.27474, 0.744444], [0.629167, 0.473264], [0.27474, 0.744444], [0.623698, 0.460417], [0.27474, 0.744444], [0.61849, 0.449306], [0.27474, 0.744792], [0.613802, 0.442708], [0.27474, 0.745139], [0.609375, 0.436111], [0.605208, 0.431597]]
#     # np_ball_coords = np.array(ransac_ball_candidates)
#     # x_vals = []
#     # y_vals = []
#     # for row in np_ball_coords:
#     #     print(row)
#     #     x_vals.append(row[0])
#     #     y_vals.append(row[1])
#     # x_vals = np.array(x_vals)
#     # y_vals = np.array(y_vals)
#     #
#     # model = np.poly1d(np.polyfit(x_vals, y_vals, 2))
#     # print(model)
#     #
#     # exit()
#
#
#     #ransac = RANSACRegressor(PolynomialFeatures(degree=2), random_state=0)
#     #reg = RANSACRegressor.fit(ransac_ball_candidates)
#     #reg = RANSACRegressor.fit(X=ransac_ball_candidates[0:5], y=ransac_ball_candidates[5:])
#
#     ball_candidates = {}
#     regression_ball_candidates = []
#
#     paths=[]
#
#     frame_num = 0
#     ret, frame = video.read()
#     # undistorting image based on camera calibration
#     frame = undistort(frame, True)
#
#     out = cv2.VideoWriter('ball_candidates_each_frame_kinematics.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
#                           (frame.shape[1], frame.shape[0]))
#
#     #looping through for now to get to ideal frame
#     for i in range(0, 63):
#         ret, frame = video.read()
#         frame_num +=1
#
#
#     #loop through rest of frames for tracking
#     while frame_num < 115:
#         print("in frame loop")
#         cv2.imshow('f', frame)
#         cv2.waitKey(0)
#         #write image so detect.py can find it
#         cv2.imwrite("current_frame.png", frame)
#
#         #code to try and zoom in but this actually resulted in more false positives
#         #if there are ball candidates from prev frame
#         # if frame_num-1 in ball_candidates.keys() and bool(ball_candidates[frame_num-1]):
#         #     ball_candidate = ball_candidates[frame_num-1][0]
#         #     dh, dw, _ = frame.shape
#         #
#         #     #center point of ball candidate
#         #     x = int(float(ball_candidate[0]) * dw)
#         #     y = int(float(ball_candidate[1]) * dh)
#         #
#         #     cv2.circle(frame, (x,y), 5, (0,0,0), 5)
#             #cv2.imshow('orig', frame)
#             #frame = frame[y-200:y+200, x-200:x+200]
#             #cv2.imshow("f", frame)
#             #cv2.waitKey(0)
#             #cv2.imwrite('current_frame.png', frame)
#
#         #call detect from yolo on frame
#         #always writes to exp, saves txt under exp/labels
#         os.system("python3.7 /Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/detect.py"
#                   " --weights '/Users/ashley20/Documents/last_neg.pt'"
#                   " --source /Users/ashley20/PycharmProjects/ThesisCameraCalibration/current_frame.png"
#                   " --save-txt"
#                   " --exist-ok")
#
#         ball_candidates[frame_num] = {}
#         #pull resulting labels of ball candidates from detect
#         with open('/Users/ashley20/PycharmProjects/ThesisCameraCalibration/yolov3/runs/detect/exp/labels/current_frame.txt',
#                   'r') as label_file:
#             #adding identifier for each ball candidate in the frame
#             label_num = 0
#             for line in label_file:
#                 line = line.split()
#                 x = float(line[1])
#                 y = float(line[2])
#                 width = float(line[3])
#                 height = float(line[4])
#                 ball_candidates[frame_num][label_num] = [x,y,width,height]
#                 label_num+=1
#                 dh, dw, _ = frame.shape
#                 x = int(float(x) * dw)
#                 y = int(float(y) * dh)
#                 regression_ball_candidates.append([x,y])
#                 cv2.circle(frame, (x,y), 5, (0,255,0), 5)
#             out.write(frame)
#
#         #cv2.imshow("f", frame)
#         #cv2.waitKey(0)
#
#         if len(paths) == 0:
#             for c in ball_candidates[frame_num]:
#                 print(c)
#                 #store array of frame_num + label_num so can go back and find it in ball_candidate dictionary
#                 paths.append([str(frame_num)+"_"+str(c)])
#                 print(paths)
#
#         else:
#             for curr in ball_candidates[frame_num]:
#                 for path in paths:
#                     #print(ball_candidates[frame_num])
#                     x_curr, y_curr, w_curr, h_curr = ball_candidates[frame_num][curr][0], ball_candidates[frame_num][curr][1], ball_candidates[frame_num][curr][2], ball_candidates[frame_num][curr][2]
#                     prev_frame_num, prev_label_num = path[0].split('_')
#                     prev_frame_num = int(prev_frame_num)
#                     prev_label_num = int(prev_label_num)
#                     x_prev, y_prev, w_prev, h_prev = ball_candidates[prev_frame_num][prev_label_num][0], ball_candidates[prev_frame_num][prev_label_num][1], ball_candidates[prev_frame_num][prev_label_num][2], ball_candidates[prev_frame_num][prev_label_num][2]
#                     print(x_curr)
#                     print(x_prev)
#                     dist = dist_formula([x_curr, y_curr], [x_prev, y_prev])
#                     direction = get_direction(ball_candidates[frame_num][curr], ball_candidates[prev_frame_num][prev_label_num])
#                     print("math info")
#                     print(frame_num, curr)
#                     print(frame_num-1, path)
#                     print(dist)
#                     print(direction)
#                     if dist > 0 and dist < 0.05:
#                         print("storing path")
#                         updated = str(frame_num) + '_' + str(curr)
#                         print(updated)
#                         path.insert(0, updated)
#                         print(path)
#
#         for path in paths:
#             print("path:")
#             print(path)
#             for i in range(len(path)):
#                 curr_frame_num, curr_label_num = path[i].split('_')
#                 dict = ball_candidates[int(curr_frame_num)][int(curr_label_num)]
#                 dh, dw, _ = frame.shape
#                 x = int(float(dict[0]) * dw)
#                 y = int(float(dict[1]) * dh)
#                 width = int(float(dict[2]) * dw)
#                 height = int(float(dict[3]) * dh)
#                 x_max = x + int(width / 2)
#                 y_max = y + int(height / 2)
#                 cv2.circle(frame, (x,y), 5, (0,0,255), 5)
#                 #regression_ball_candidates.append([x,y])
#
#         for all_bc in regression_ball_candidates:
#             cv2.circle(frame, (all_bc[0], all_bc[1]), 5, (0, 255, 0), 5)
#
#
#         #out.write(frame)
#         if frame_num == 96:
#             cv2.imshow("frame", frame)
#             cv2.imwrite("all_ball_candidates_detector_only.png", frame)
#             cv2.waitKey(0)
#
#         #go to next frame
#         ret, frame = video.read()
#         frame_num+=1
#
#     #print(ransac_ball_candidates)
#     print(ball_candidates)
#     print(paths)
#     out.release()
#
#     "print long paths"
#     for path in paths:
#         if len(path) > 3:
#             print(path)
