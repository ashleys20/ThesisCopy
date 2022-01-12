#code adapted from https://github.com/niconielsen32/ComputerVision/blob/master/StereoVision/Python/HSV_filter.py

import numpy as np
import cv2
import imutils

# currenlty, frames that are sent as args are already distorted
# this function looks to detect lines of the field to check if out of bounds
def detection(frame_left, frame_right):
    kernel_size = 5
    mag_thresh = (30, 100)
    r_thresh = (235, 255)
    s_thresh = (165, 255)
    b_thresh = (160, 255)
    g_thresh = (210, 255)
    #combined_binary = get_bin_img(frame_left, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh,
                                  #s_thresh=s_thresh, b_thresh=b_thresh, g_thresh=g_thresh)

    #cv2.imshow("window_binary", combined_binary)
    #cv2.waitKey(0)
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    #f.tight_layout()
    #ax1.imshow(cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
    #ax1.set_title("Original:: ", fontsize=18)
    #ax2.imshow(combined_binary, cmap='gray')
    #ax2.set_title("Threshold Binary:: ", fontsize=18)
    #f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")

    find_field_lines(frame_left)
    find_field_lines(frame_right)

    cv2.imwrite("frame_left.png", frame_left)
    cv2.imwrite("frame_right.png", frame_right)

    mask_left = hsv_filter_yellow(frame_left)
    mask_right = hsv_filter_yellow(frame_right)

    # Result-frames after applying HSV-filter mask
    updated_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)
    updated_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)

    #cv2.imshow('image', frame_left)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # APPLYING SHAPE RECOGNITION:
    # this function returns center coordinate of shot put
    circle_left = find_shot(frame_left, mask_left)
    circle_right = find_shot(frame_right, mask_right)

    print(circle_left)
    print(circle_right)

    circle_color = (255, 0, 0)
    cv2.circle(frame_left, circle_left, 10, circle_color, 2)
    cv2.circle(frame_right, circle_right, 10, circle_color, 2)

    cv2.imshow("image", frame_left)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("image2", frame_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return circle_left, circle_right


def hsv_filter_yellow(frame):
    # Blurring the frame
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # Converting RGB to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #code for determining lower and upper limits:
    #https://docs.opencv.org/3.1.0/df/d9d/tutorial_py_colorspaces.html
    yellow = np.uint8([[[255, 255, 255]]])
    hsv_yellow = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)

    #code for segmentating the yellow spike ball in the image
    #lower_yellow = np.array([hsv_yellow[0][0][0] - 10, 100, 100])
    #upper_yellow = np.array([hsv_yellow[0][0][0] - 10, 255, 255])

    lower_yellow = np.array([15, 75, 75])
    upper_yellow = np.array([30, 255, 255])

    print("lower yellow: ")
    print(lower_yellow)
    print("upper yellow")
    print(upper_yellow)

    # HSV-filter mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    #cv2.imshow("image", mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Morphological Operation - Opening - Erode followed by Dilate - Remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow("window", mask)
    cv2.waitKey(0)

    return mask

def hsv_filter_white(frame):
    # Blurring the frame
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # Converting RGB to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    cv2.imshow("window", hsv)
    cv2.waitKey(0 )

    sensitivity = 135
    lower_white = np.array([0, 0, 255-sensitivity], dtype=np.uint8)
    upper_white = np.array([255, sensitivity, 255], dtype=np.uint8)

    # HSV-filter mask
    mask = cv2.inRange(hsv, lower_white, upper_white)

    #cv2.imshow("image", mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Morphological Operation - Opening - Erode followed by Dilate - Remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask

#returns array of center points of circles found when shot is sitting on ground
def find_shot(frame, mask):

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None

    if len(contours) <= 0: print("no contours")

    # Only proceed if at least one contour was found
    if len(contours) > 0:
        print("contours found")
        # Find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        print("num contours")
        print(len(contours))
        #c = max(contours, key=cv2.contourArea)

        center_points = []

        for c in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)  # Finds center point
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                print(center)
                print(radius)

                # Only proceed if the radius is greater than a minimum value
                if radius > 2 and radius < 20:
                    print("radius > 2")
                    # Draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255, 0), 2)
                    cv2.circle(frame, center, 5, (0, 0,  0), -1)
                    #adding detected center coordinate to list of points
                    center_points.append(center)
    print("center points:")
    print(len(center_points))
    return np.array(center_points)

#Code adapted from:
#https://www.tutorialspoint.com/line-detection-in-python-with-opencv
def find_field_lines(frame):
    #added filter for white to help find correct lines
    mask = hsv_filter_white(frame)
    updated_frame = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("window", mask)
    cv2.waitKey(0)
    cv2.imshow("window", updated_frame)
    cv2.waitKey(0)

    #convert to grayscale
    gray = cv2.cvtColor(updated_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("window", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #increased the maximum value for threshold to get cleaner results, eliminates lots of noise
    #need aperturesize to be the minimum
    #set L2 gradient to true for more accurate L2 norm to clean up number of lines
    edges = cv2.Canny(gray, 10, 400, apertureSize=3, L2gradient=True)

    #finds line segments in binary image using probabilistic Hough transform
    #as opposed to HoughLines() which uses standard Hough transform
    #increased threshold to reduce number of detected lines
    #lowered maxLineGap
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, maxLineGap=75, minLineLength=200)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (x2 - x1) != 0:
            slope = (y2 - y1) / (x2 - x1)
        #print(slope)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("linesEdges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("linesDetected", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#https://github.com/nachiket273/Self_Driving_Car/tree/master/CarND-Advanced-Lane-Lines
#perspective transform maps the points in given image to diff perspective
#goal is to get lines of shot put field to appear parallel
def perspective_transform(frame, offset = 250):
    img = frame
    img_size = (img.shape[1], img.shape[0])

    out_img_orig = np.copy(img)

    leftupper = (585, 460)
    rightupper = (705, 460)
    leftlower = (210, img.shape[0])
    rightlower = (1080, img.shape[0])

    warped_leftupper = (offset, 0)
    warped_rightupper = (offset, img.shape[0])
    warped_leftlower = (img.shape[1] - offset, 0)
    warped_rightlower = (img.shape[1] - offset, img.shape[0])

    color_r = [0, 0, 255]
    color_g = [0, 255, 0]
    line_width = 5

    src = np.float32([leftupper, leftlower, rightupper, rightlower])
    dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])

    cv2.line(out_img_orig, leftlower, leftupper, color_r, line_width)
    cv2.line(out_img_orig, leftlower, rightlower, color_r, line_width * 2)
    cv2.line(out_img_orig, rightupper, rightlower, color_r, line_width)
    cv2.line(out_img_orig, rightupper, leftupper, color_g, line_width)

    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    out_warped_img = np.copy(warped)

    cv2.line(out_warped_img, warped_rightupper, warped_leftupper, color_r, line_width)
    cv2.line(out_warped_img, warped_rightupper, warped_rightlower, color_r, line_width * 2)
    cv2.line(out_warped_img, warped_leftlower, warped_rightlower, color_r, line_width)
    cv2.line(out_warped_img, warped_leftlower, warped_leftupper, color_g, line_width)

    return warped, M, minv, out_img_orig, out_warped_img


def abs_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), return_grad=False, direction='x'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad = None
    scaled_sobel = None

    # Sobel x
    if direction.lower() == 'x':
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
    # Sobel y
    else:
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # Take the derivative in y

    if return_grad == True:
        return grad

    abs_sobel = np.absolute(grad)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1

    return grad_binary


def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    xgrad = abs_thresh(img, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh, return_grad=True)
    ygrad = abs_thresh(img, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh, return_grad=True, direction='y')

    magnitude = np.sqrt(np.square(xgrad) + np.square(ygrad))
    abs_magnitude = np.absolute(magnitude)
    scaled_magnitude = np.uint8(255 * abs_magnitude / np.max(abs_magnitude))
    mag_binary = np.zeros_like(scaled_magnitude)
    mag_binary[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude < mag_thresh[1])] = 1

    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    xgrad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    ygrad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    xabs = np.absolute(xgrad)
    yabs = np.absolute(ygrad)

    grad_dir = np.arctan2(yabs, xabs)

    binary_output = np.zeros_like(grad_dir).astype(np.uint8)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir < thresh[1])] = 1
    return binary_output


def get_rgb_thresh_img(img, channel='R', thresh=(0, 255)):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channel == 'R':
        bin_img = img1[:, :, 0]
    if channel == 'G':
        bin_img = img1[:, :, 1]
    if channel == 'B':
        bin_img = img1[:, :, 2]

    binary_img = np.zeros_like(bin_img).astype(np.uint8)
    binary_img[(bin_img >= thresh[0]) & (bin_img < thresh[1])] = 1

    return binary_img


def get_hls_lthresh_img(img, thresh=(0, 255)):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = hls_img[:, :, 1]

    binary_output = np.zeros_like(L).astype(np.uint8)
    binary_output[(L >= thresh[0]) & (L < thresh[1])] = 1

    return binary_output


def get_hls_sthresh_img(img, thresh=(0, 255)):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls_img[:, :, 2]

    binary_output = np.zeros_like(S).astype(np.uint8)
    binary_output[(S >= thresh[0]) & (S < thresh[1])] = 1

    return binary_output


def get_lab_athresh_img(img, thresh=(0, 255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = lab_img[:, :, 1]

    bin_op = np.zeros_like(A).astype(np.uint8)
    bin_op[(A >= thresh[0]) & (A < thresh[1])] = 1

    return bin_op


def get_lab_bthresh_img(img, thresh=(0, 255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    B = lab_img[:, :, 2]

    bin_op = np.zeros_like(B).astype(np.uint8)
    bin_op[(B >= thresh[0]) & (B < thresh[1])] = 1

    return bin_op


def get_bin_img(img, kernel_size=3, sobel_dirn='X', sobel_thresh=(0, 255), r_thresh=(0, 255),
                s_thresh=(0, 255), b_thresh=(0, 255), g_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if sobel_dirn == 'X':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    combined = np.zeros_like(sbinary)
    combined[(sbinary == 1)] = 1

    # Threshold R color channel
    r_binary = get_rgb_thresh_img(img, thresh=r_thresh)

    # Threshhold G color channel
    g_binary = get_rgb_thresh_img(img, thresh=g_thresh, channel='G')

    # Threshhold B in LAB
    b_binary = get_lab_bthresh_img(img, thresh=b_thresh)

    # Threshold color channel
    s_binary = get_hls_sthresh_img(img, thresh=s_thresh)

    # If two of the three are activated, activate in the binary image
    combined_binary = np.zeros_like(combined)
    combined_binary[(r_binary == 1) | (combined == 1) | (s_binary == 1) | (b_binary == 1) | (g_binary == 1)] = 1

    return combined_binary

#once it knows where the ball is, can detect if ball is in bounds or not based on knowing where field lines are
def out_of_bounds(frame, shot_center):
    if shot_center is None:
        print("no implement detected")
        return

    #print("in out of bounds")
    image_size = frame.shape
    print("image size ", image_size)
    image_width = image_size[1]
    print("image width ", image_width)
    image_height = image_size[0]
    print("image height ", image_height)
    shot_width = shot_center[0]
    print("shot width ", shot_width)
    shot_height = shot_center[1]
    print("shot height ", shot_height)

    #move to right until detect red line
    curr_width = shot_width
    curr_pixel = shot_center
    while curr_width < image_width:
        curr_pixel_rgb = frame[curr_pixel[1], curr_pixel[0]]
        if np.array_equal(curr_pixel_rgb, [0, 0, 255]):
            print("curr pixel", curr_pixel)
            #cv2.circle(frame, (curr_pixel[0], curr_pixel[1]), 5,
                       #(0, 255, 255), 2)
            print("in bounds on right side")
            break
        curr_width = curr_pixel[0] + 1
        curr_pixel = [curr_width, curr_pixel[1]]

    #if height has reached end of image, then it didnt detect line and ball is out of bounds
    if curr_width == image_width:
        print("out of bounds on left")
        return True

    #move to left until detect red line
    curr_width = shot_width
    curr_pixel = shot_center
    while 0 <= image_width:
        curr_pixel_rgb = frame[curr_pixel[1], curr_pixel[0]]
        if np.array_equal(curr_pixel_rgb, [0, 0, 255]):
            print("curr pixel", curr_pixel)
            #cv2.circle(frame, (curr_pixel[0], curr_pixel[1]), 5,
                       #(0, 255, 255), 2)
            print("in bounds on left side")
            break
        curr_width = curr_pixel[0] + 1
        curr_pixel = [curr_width, curr_pixel[1]]

    # if height has reached end of image, then it didnt detect line and ball is out of bounds
    if curr_width == image_width:
        print("out of bounds on left")
        return True

    return False

#code to find stop board which is where measurement will take place
def find_stop_board(frame):
    # added filter for white to help find correct lines
    mask = hsv_filter_white(frame)
    updated_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # convert to grayscale
    gray = cv2.cvtColor(updated_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("find stop board grayscale", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # get canny edges
    # increased the maximum value for threshold to get cleaner results, eliminates lots of noise
    # need aperturesize to be the minimum
    # set L2 gradient to true for more accurate L2 norm to clean up number of lines
    edges = cv2.Canny(gray, 10, 400, apertureSize=3, L2gradient=True)

    #also look at cv2.CHAIN_APPROX_NONE
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) <= 0: print("no contours")

    # Only proceed if at least one contour was found
    if len(contours) > 0:
        print("contours found")
        print("num contours")
        print(len(contours))
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        cv2.imshow('Contours', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#hough circles worked okay but better to use findContours because it is more robust
#and generalizes well to different images
def find_shot_hough_circles(frame):
    # testing Hough circles
    print("in hough circles")
    output = frame.copy()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("in find shot")
    #mindist can't exactly be set to a certain value b/c won't guarantee shot gets detected
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1.2, minDist=500, maxRadius=100)
    if circles is None:
        print("no circles found")
    else:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print("in for loop")
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
        cv2.imshow("output", np.hstack([frame, output]))
        cv2.waitKey(0)
