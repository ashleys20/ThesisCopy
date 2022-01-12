import cv2
import pandas as pd
import glob

ball_coord_data = []
cropping = False

#source https://www.life2coding.com/crop-image-using-mouse-click-movement-python/
def mouse_crop(event, x, y, flags, param):
    #print("in mouse crop")
    video_name = param[0]
    frame_name = param[1]
    cropped_frame_name = param[2]
    frame = param[3]
    count = param[4]

    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        print("button down")
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        print("button up")
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False
        refPoint = [(x_start, y_start), (x_end, y_end)]
        print(refPoint)
        if len(refPoint) == 2: #when two points were found
            roi = frame[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped" + str(count), roi)
            cv2.waitKey(2000)
            cv2.destroyWindow("Cropped" + str(count))
            #save cropped image to folder
            cv2.imwrite("Cropped Ball Images/" + cropped_frame_name, roi)
            # save original images to Frames folder
            cv2.imwrite('Frames/' + frame_name, frame)

            #get center ball coordinates, midpoint formula
            ball_x = int((x_start + x_end)/2)
            ball_y = int((y_start + y_end)/2)

            #append to ball_coord_data
            ball_coord_data.append([video_name, frame_name, cropped_frame_name, ball_x, ball_y])
            print(ball_coord_data)
            # put ball coord data into data frame and then write csv file
            df = pd.DataFrame(ball_coord_data, columns=['video_name', 'frame_name', 'cropped_img_name', 'ball_x', 'ball_y'])
            df.to_csv('ball_coord_data9.csv', index=False)



def collect_data(isLeft):
    if isLeft is True: folder =  '/Users/ashley20/PycharmProjects/ThesisCameraCalibration/Thesis_Data_Videos_Left/veryclose_1_24_behind_spike_on_190_left.MP4'
    else: folder = '/Users/ashley20/PycharmProjects/ThesisCameraCalibration/Thesis_Data_Videos_Right/throwoutofbounds_2_292_behind_shot_on_160_right.MP4'

    for filepath in glob.glob(folder):
        # read video
        video = cv2.VideoCapture(filepath)
        success, frame = video.read()
        count = 0
        if isLeft: video_name = filepath[80:]
        else: video_name = filepath[81:]
        print(video_name)
        # loop through frames
        while success:
            count += 1
            # for each frame, allow manual cropping
            frame_name = video_name + "_" + str(count) + ".png"
            #print(frame_name)
            cropped_frame_name = "ball_" + video_name + "_" + str(count) + ".png"
            #print(cropped_frame_name)


            #cv2.namedWindow("curr frame")
            cv2.imshow("curr frame" + str(count), frame)

            params = [video_name, frame_name, cropped_frame_name, frame, count]
            cv2.setMouseCallback("curr frame" + str(count), mouse_crop, params)
            cv2.waitKey(0)
            cv2.destroyWindow("curr frame" + str(count))
            # reset image for next iteration
            success, frame = video.read()


if __name__ == '__main__':
    #print("in data collection")
    #isLeft = True
    #collect_data(isLeft)
    isLeft = False
    collect_data(isLeft)