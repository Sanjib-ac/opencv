import cv2
import numpy as np


# path = r'C:\Users\Public\Documents\opencv-project\files\shapes.png'
# img = cv2.imread(path)
# imageContour = img.copy()


def auto_canny_edge_detection(image, sigma=0.33):
    md = np.median(image)
    lower_value = int(max(0, (1.0 - sigma) * md))
    upper_value = int(min(255, (1.0 + sigma) * md))
    return cv2.Canny(image, lower_value, upper_value)


def getContoures(src_img, des):
    # contours, hierarchy = cv2.findContours(src_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(src_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(des, cnt, -1, (1, 1, 1), 3)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.021 * peri, True)
            objCor = len(approx)
            # print(objCor)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor == 3:
                objectType = 'Tri'
            elif 6 < objCor < 10:
                if objCor < 8:
                    objectType = 'quater'
                else:
                    objectType = 'half'
            else:
                objectType = 'None'
            cv2.rectangle(des, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(des, objectType,
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 0), 2)


# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
# imgCanny = cv2.Canny(imgBlur, 50, 50)
# getContoures(imgCanny, imageContour)
#
# cv2.imshow('Original', imageContour)
# # cv2.imshow('Original1', img)
# cv2.waitKey(0)

#

################################
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    read_ok_g, frame_g = cap.read()
    frame_gg = frame_g.copy()
    imgGray = cv2.cvtColor(frame_g, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    # ret, thresh = cv2.threshold(imgBlur, 50, 120, cv2.THRESH_BINARY)
    # imgBlur = cv2.GaussianBlur(imgGray, (7, 7), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
    imgCanny = cv2.Canny(imgBlur, 40, 120)
    # imgCanny = auto_canny_edge_detection(imgBlur)
    # edged = cv2.dilate(imgCanny, None, iterations=1)
    # edged = cv2.erode(edged, None, iterations=1)
    getContoures(imgCanny, frame_gg)
    cv2.imshow("GPU", frame_gg)

    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
######################################################
# import numpy as np
# import cv2 as cv
#
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# while True:
#     read_ok_g, frame_g = cap.read()x
#     img = cv.medianBlur(frame_g, 5)
#     cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     circles = cv.HoughCircles(cimg, cv.HOUGH_GRADIENT, 1, 20,
#                               param1=50, param2=30, minRadius=0, maxRadius=0)
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         # draw the outer circle
#         cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         # draw the center of the circle
#         cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
#         cv.imshow('detected circles', cimg)
#     if cv2.waitKey(1) & 0xFF == ord('x'):
#         break
#     cv.destroyAllWindows()
