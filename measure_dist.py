#class to measure the distance once the pixel coordinates of the shot put are known
import cv2
from camera_calibration import undistort
import numpy as np
import skimage.transform as skt

#pixel coordinates of center of stop box


def normalize_coordinates(arr, img):
    num_rows, num_cols = img.shape[:2]
    new_arr = []
    for val in arr:
        x = float(val[0]) / float(num_cols - 1.0)
        y = float(float(val[1]) / float(num_rows - 1.0))
        new_arr.append((x,y))

    return np.array(new_arr)

def denormalize_coordinates(arr, img):
    num_rows, num_cols = img.shape[:2]
    new_arr = []
    for val in arr:
        x = float(val[0]) * float(num_cols - 1.0)
        y = float(float(val[1]) * float(num_rows - 1.0))
        new_arr.append((int(x), int(y)))

    return np.array(new_arr)

def click_event_left(event, x, y, flags, params):
    #print(params[0])
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(params[0], str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('curr frame', params[0])
        eight_coords_left.append((x,y))

def click_event_right(event, x, y, flags, params):
    print(params[0])
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(params[0], str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('curr frame', params[0])
        eight_coords_right.append((x,y))


def get_frames():
    #get initial frame without any thrower in it
    calib_video_left = cv2.VideoCapture('Thesis_Data_Videos_Left/court_2_292_behind_on_none_190_left.MP4')
    calib_video_right = cv2.VideoCapture('Thesis_Data_Videos_Right/court_2_292_behind_on_none_190_right.MP4')

    success, frame_left = calib_video_left.read()
    #frame_left = undistort(frame_left, isLeft=True)
    success, frame_right = calib_video_right.read()
    #frame_right = undistort(frame_right, isLeft=False)
    return frame_left, frame_right

    cv2.imshow('curr frame', frame_left)
    params = [frame_left]
    cv2.setMouseCallback("curr frame", click_event_left, params)
    cv2.waitKey(0)
    cv2.destroyWindow("curr frame")

    cv2.imshow('curr frame', frame_right)
    params = [frame_right]
    cv2.setMouseCallback("curr frame", click_event_right, params)
    cv2.waitKey(0)
    cv2.destroyWindow("curr frame")

    print(eight_coords_left)
    print(eight_coords_right)
    return frame_left, frame_right

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def compute_fund_matrix(frame_left, frame_right, eight_coords_left, eight_coords_right):
    F, mask = cv2.findFundamentalMat(eight_coords_left, eight_coords_right, cv2.FM_8POINT)
    print(F)
    eight_coords_left = denormalize_coordinates(eight_coords_left, frame_left)
    eight_coords_right = denormalize_coordinates(eight_coords_right, frame_right)

    lines1 = cv2.computeCorrespondEpilines(eight_coords_right.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    img5, img6 = drawlines(frame_left, frame_right, lines1, eight_coords_left, eight_coords_right)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(eight_coords_left.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(frame_right, frame_left, lines2, eight_coords_right, eight_coords_left)

    cv2.imshow('img5', img5)
    cv2.imshow('img6', img6)
    cv2.imshow('img3', img3)
    cv2.imshow('img4', img4)
    cv2.waitKey(0)
    return F

#test with Harris corner detector - gets lots of corners up top but fails to identify the throwing ring or measure box
def get_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    print(corners)
    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    frame[res[:, 1], res[:, 0]] = [0, 0, 255]
    frame[res[:, 3], res[:, 2]] = [0, 255, 0]
    cv2.imwrite('subpixeltest.png', frame)


#click for the 8 points

#save points

#calculate fundamental matrix


if __name__ == '__main__':
   print("in measure_dist")
   frame_left, frame_right = get_frames()
   frame_left = undistort(frame_left, True)
   frame_right = undistort(frame_right, False)
   eight_coords_left_undistorted = np.array([(1324,1217), (787,903), (1341,947), (950,754), (1252,782), (477,563), (1545,593), (1061,462), (1665,1319), (1099,1359), (970,1198), (1576,1190)])
   eight_coords_right_undistorted = np.array([(671,957), (589,655), (1180,698), (848,500), (1170,523), (109,276), (1485,300), (873,146), (934,1114), (206,1041), (322,906), (1056,967)])
   eight_coords_left_distorted = np.array([(1340,1223), (896,953), (1360,964), (1031,812), (1285,817), (709,704), (1563,638), (1147,564)])
   eight_coords_right_distorted = np.array([(803,1004), (773,761), (1224,746), (975,610), (1230,600), (533,549), (1518,405), (1034,355)])

   #normalize coordinates by dividing everything by dimensions
   eight_coords_left = normalize_coordinates(eight_coords_left_undistorted, frame_left)
   #print(eight_coords_left)
   eight_coords_right = normalize_coordinates(eight_coords_right_undistorted, frame_right)

   #Step 1 - compute 3x3 fundamental matrix from eight points in camera
   F = compute_fund_matrix(frame_left, frame_right, eight_coords_left, eight_coords_right)

   #Step 2 - compute camera matrices P and P'
   #P is identity matrix plus column of 0's
   #P = np.append(np.identity(3), np.zeros((3,1)), axis=1)
   #print(P)
   #P_prime is e'.T + F + e' but how do I actually get e', all I have is F and epilines

   #can go fundamental --> essential --> rotation and translation matrices (4 solutions but only one will be valid) (this is called pose recovery)
   # --> projection matrix --> 3D coordinates
   #essential = K'.T * fundamental * K where K is intrinsic matrix
   K_left = np.load('left_intrinsic_matrix.npy')
   K_right = np.load('right_intrinsic_matrix.npy')
   print("K left:")
   print(K_left)
   print("K right:")
   print(K_right)
   essential_matrix = np.transpose(K_right) * F * K_left
   print("essential matrix:")
   print(essential_matrix)

   U, S, V = np.linalg.svd(essential_matrix)
   print(U)
   print(S)
   print(V)








