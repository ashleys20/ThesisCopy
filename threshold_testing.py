# import cv2
#
# ret, binarized_mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
#     cv2.imshow("thresh1", binarized_mask)
#     cv2.waitKey(0)
#
#     th2 = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#     th3 = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#
#     cv2.imshow("thresh2", th2)
#     cv2.waitKey(0)
#     cv2.imshow("thresh3", th3)
#     cv2.waitKey(0)
#
#     ret2, th4 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imshow("thresh4", th4)
#     cv2.waitKey(0)
#
#     #increased window for blur
#     blur = cv2.GaussianBlur(mask, (13, 13), 0)
#     ret3, th5 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imshow("thresh5", th5)
#     cv2.waitKey(0)