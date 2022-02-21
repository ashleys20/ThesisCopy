#class to measure the distance once the pixel coordinates of the shot put are known
import cv2
from camera_calibration import undistort
import numpy as np
import skimage.transform as skt
from scipy import linalg
from formulas import dist_formula

#pixel coordinates of center of stop box

eight_coords_left = []
eight_coords_right = []

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
    calib_video_left = cv2.VideoCapture('Thesis_Data_Videos_Left/throwfar_2_292_behind_shot_on_190_left.MP4')
    calib_video_right = cv2.VideoCapture('Thesis_Data_Videos_Right/throwfar_2_292_behind_shot_on_190_right.MP4')

    frame_num=0
    for x in range(65):
        success, frame_left = calib_video_left.read()
        frame_num +=1
    frame_left = undistort(frame_left, True)
    #cv2.imshow("f", frame_left)
    #cv2.waitKey(0)
    #print(frame_num)
    #frame_left = undistort(frame_left, isLeft=True)
    frame_num = 0
    for x in range(68):
        success, frame_right = calib_video_right.read()
        frame_num += 1
    frame_right = undistort(frame_right, False)
    #cv2.imshow("f", frame_right)
    #cv2.waitKey(0)
    #print(frame_num)
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

    #print(eight_coords_left)
    #print(eight_coords_right)
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
    #eight_coords_left = denormalize_coordinates(eight_coords_left, frame_left)
    #eight_coords_right = denormalize_coordinates(eight_coords_right, frame_right)

    lines1 = cv2.computeCorrespondEpilines(eight_coords_right.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    img5, img6 = drawlines(frame_left, frame_right, lines1, eight_coords_left, eight_coords_right)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(eight_coords_left.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(frame_right, frame_left, lines2, eight_coords_right, eight_coords_left)

    # cv2.imshow('img5', img5)
    # cv2.imshow('img6', img6)
    # cv2.imshow('img3', img3)
    # cv2.imshow('img4', img4)
    # cv2.waitKey(0)
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

#Source: https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914
def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]

def triangulate_points(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2 (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    X = [triangulate_nviews([P1, P2], [x[0], x[1]]) for x in zip(x1, x2)]
    return np.array(X)

# essential = K'.T * fundamental * K where K is intrinsic matrix left camera, K' is intrinsic matrix right camera
def find_essential_matrix(F, K_left, K_right):
    essential_matrix = np.transpose(K_right) @ F @ K_left
    return essential_matrix



#can go fundamental --> essential --> rotation and translation matrices (4 solutions but only one will be valid) (this is called pose recovery)
# --> projection matrix --> 3D coordinates
if __name__ == '__main__':
   print("in measure_dist")

   #get frames and click for corresponding points here
   frame_left, frame_right = get_frames()

   #get intrinsic matrices
   K_left = np.load('left_intrinsic_matrix.npy')
   K_right = np.load('right_intrinsic_matrix.npy')

   #eight_coords_left_undistorted = np.array([(1324,1217), (787,903), (1341,947), (950,754), (1252,782), (477,563), (1545,593), (1061,462), (1665,1319), (1099,1359), (970,1198), (1576,1190)])
   #eight_coords_right_undistorted = np.array([(671,957), (589,655), (1180,698), (848,500), (1170,523), (109,276), (1485,300), (873,146), (934,1114), (206,1041), (322,906), (1056,967)])
   #eight_coords_left_distorted = np.array([(1340,1223), (896,953), (1360,964), (1031,812), (1285,817), (709,704), (1563,638), (1147,564)])
   #eight_coords_right_distorted = np.array([(803,1004), (773,761), (1224,746), (975,610), (1230,600), (533,549), (1518,405), (1034,355)])
   eight_coords_left_test = np.array([(1016, 1198), (384, 851), (1028, 925), (587, 702), (932, 757), (552, 727), (983, 545), (1124, 574), (1243, 601), (1342, 565), (1488, 517), (727, 411)])
   eight_coords_right_test = np.array([(779, 960), (707, 668), (1274, 702), (954, 512), (1267, 530), (930, 543), (1292, 268), (1443, 283), (1580, 292), (1673, 229), (1820, 129), (981, 164)])

   #normalize coordinates by dividing everything by dimensions
   #eight_coords_left = normalize_coordinates(eight_coords_left_undistorted, frame_left)
   #eight_coords_right = normalize_coordinates(eight_coords_right_undistorted, frame_right)

   #eight_coords_left = np.array(eight_coords_left)
   #eight_coords_right = np.array(eight_coords_right)

   #STEP 1 - compute 3x3 fundamental matrix from eight or more points in camera
   F = compute_fund_matrix(frame_left, frame_right, eight_coords_left_test, eight_coords_right_test)

   # STEP 2 - compute essential matrix
   E = find_essential_matrix(F, K_left, K_right)
   #print("essential matrix:")
   #print(essential_matrix)

   #STEP 3 - extract rotation and translation matrices

   #first do SVD
   U, S, V = np.linalg.svd(E)
   # there will be slight error because the bottom right value of S is not exactly zero even though it is very close
   S = np.diag(S)
   V_transpose = np.transpose(V)
   #print("U:", U)
   #print("S:", S)
   #print("V:", V)
   #print("V transpose:", V_transpose)

   #construct W, W_t and Z
   W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
   W_transpose = np.transpose(W)
   Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

   #translation = U*Z*U_T or -(U*Z*U_T)
   #rotation = U*W*V_T or U*W_T*V_T
   translation_solution = U @ Z @ np.transpose(U)
   neg_translation_solution = -1 * translation_solution
   rot_solution_1 = U @ W @ V_transpose
   rot_solution_2 = U @ W_transpose @ V_transpose

   #STEP 4: Construct four possible projection matrix solutions

   #getting u and -u, which is the last column of U
   u = np.dot(U, np.array([[0], [0], [1]], order='F'))
   neg_u = -1 * u
   print(u)

   # 4 possible solutions for P_right = K*[R|t] because the direction of the translation vector could be reversed
   # and the two rotation soltuions represent a rotation of 180 degrees about the line joining the two camera centers - "twisted pair"
   # two possible directions and two possible roations = 4 possible solutions
   P21 = K_right @ np.concatenate((rot_solution_1, u), axis=1)
   P22 = K_right @ np.concatenate((rot_solution_1, neg_u), axis=1)
   P23 = K_right @ np.concatenate((rot_solution_2, u), axis=1)
   P24 = K_right @ np.concatenate((rot_solution_2, neg_u), axis=1)

   print("P21:")
   print(P21)
   print("P22:")
   print(P22)
   print("P23:")
   print(P23)
   print("P24:")
   print(P24)

   #STEP 5 - figure out which solution is correct

   #P1 is identity matrix plus column of 0's
   #the positive depth constraint allows one to disambiguate the impossible solutions from the one true solution
   P1 = K_left @ np.append(np.identity(3), np.zeros((3,1)), axis=1)
   print("P1:", P1)

   #for testing purposes
   stop_box_coords_left = np.array((1017,1200))
   stop_box_coords_right = np.array((781,960))
   shot_put_coords_left = np.array((833,967))
   shot_put_coords_right = np.array((965, 740))

   stop_box_coords_left_h = np.array((1017, 1200, 1))
   stop_box_coords_right_h = np.array((781, 960, 1))
   shot_put_coords_left_h = np.array((833, 967, 1))
   shot_put_coords_right_h = np.array((965, 740, 1))

   stop_box_coords = np.array((stop_box_coords_left_h, stop_box_coords_right_h))
   shot_put_coords = np.array((shot_put_coords_left_h, shot_put_coords_right_h))

   #triangulate with each projection matrix, then run it back with the projection matrix to get camera coords
   global_1 = np.array(triangulate_nviews([P1,P21], [stop_box_coords_left_h, stop_box_coords_right_h]))
   print("1:")
   print(global_1)
   x1_1 = P1 @ global_1
   x2_1 = P21 @ global_1
   print(x1_1)
   print(x2_1)
   if x1_1[2] > 0 and x2_1[2] > 0:
       final_right_proj_matrix = P21

   global_2 = triangulate_nviews([P1, P22], [stop_box_coords_left_h, stop_box_coords_right_h])
   print("2:")
   print(global_2)
   x1_2 = P1 @ global_2
   x2_2 = P22 @ global_2
   print(x1_2)
   print(x2_2)
   if x1_2[2] > 0 and x2_2[2] > 0:
       final_right_proj_matrix = P22


   global_3 = triangulate_nviews([P1, P23], [stop_box_coords_left_h, stop_box_coords_right_h])
   print("3:")
   print(global_3)
   x1_3 = P1 @ global_3
   x2_3 = P23 @ global_3
   print(x1_3)
   print(x2_3)
   if x1_3[2] > 0 and x2_3[2] > 0:
       final_right_proj_matrix = P23

   global_4 = triangulate_nviews([P1, P24], [stop_box_coords_left_h, stop_box_coords_right_h])
   print("4:")
   print(global_4)
   x1_4 = P1 @ global_4
   x2_4 = P24 @ global_4
   print(x1_4)
   print(x2_4)
   if x1_4[2] > 0 and x2_4[2] > 0:
       final_right_proj_matrix = P24

   #STEP 6 - get global coordinates using correct projection matrix

   global_stopbox = triangulate_nviews([P1, final_right_proj_matrix], [stop_box_coords_left_h, stop_box_coords_right_h])
   global_shotput = triangulate_nviews([P1, final_right_proj_matrix], [shot_put_coords_left_h, shot_put_coords_right_h])
   print("gst:", global_stopbox)
   print("gsp:", global_shotput)

   #STEP 7 - calculate distance between the two using x&z coordinates since y does not matter in measurement
   dist = dist_formula((global_stopbox[0], global_stopbox[2]), (global_shotput[0], global_shotput[2]))
   print("dist: ", dist)












