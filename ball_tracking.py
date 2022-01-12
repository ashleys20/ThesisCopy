import cv2
import numpy as np
import find_implement

#testing out built in opencv trackers, they struggle to follow the ball as it leaves
#the throwers hand and as it in traveling through the air
def track_shot_put(tracker_type):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.legacy.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()

    video = cv2.VideoCapture('Thesis_Data_Videos_Left/throwfar_2_292_behind_shot_on_160_left.MP4')

    if not video.isOpened():
        print("no video opened")
        exit()

    ret, frame = video.read()
    if not ret:
        print("no frame read")
        exit()

    frame_num = 1
    while(frame_num < 54):
        ret, frame = video.read()
        frame_num +=1
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)
    bbox = cv2.selectROI(frame, False)

    ret = tracker.init(frame, bbox)

    while True:
        ret, frame = video.read()
        if not ret: break

        timer = cv2.getTickCount()
        ret, bbox = tracker.update(frame)

        #calculate frames per second
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ret:
        # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
        # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(frame, tracker_type + "Tracker", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow("Tracking", frame)
        cv2.waitKey(0)

#test function to look at background subtraction between MOG2 and KNN
def background_subtraction():
    video = cv2.VideoCapture('Thesis_Data_Videos_Left/throwfar_2_292_behind_shot_on_160_left.MP4')

    if not video.isOpened():
        print("no video opened")
        exit()

    #here, played around with dist threshold paramaters to get ideal balance
    #between detecting trivial motions and being able to keep track of the ball
    #made detect shadows true in case helps with detecting traveling of ball
    #KNN appears cleaner than MOG2
    backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=3000, detectShadows=True)
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=250, detectShadows=True)

    while True:
        ret, frame = video.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)
        print(fgMask)
        #fgMask = cv2.GaussianBlur(fgMask, (5, 5), 0)
        #fgMask = cv2.erode(fgMask, None, iterations=2)
        #fgMask = cv2.dilate(fgMask, None, iterations=2)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(video.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        #observation - right when ball is launched, it is most obvious to detect it is circle
        #at highest velocity
        #cv2.imshow('Frame', frame)
        #cv2.imshow('FG Mask', fgMask)

        #use mask to look for ball
        circle = find_implement.find_shot(frame, fgMask)
        circle_color = (0, 0, 255)
        #cv2.circle(frame, circle, 10, circle_color, 2)
        cv2.imshow("frame with circle", frame)

        keyboard = cv2.waitKey(0)
        if keyboard == 'q' or keyboard == 27:
            break