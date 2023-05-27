# import the necessary packages
import numpy as np
import argparse
import cv2
import time

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Set Capture Device,

# print(f'camera fps = {cap.get(5)}')
# Set Width and Height
# cap.set(3,1280)
# cap.set(4,720)

# The above step is to set the Resolution of the Video. The default is 640x480.
# This example works with a Resolution of 640x480.
time_start = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # load the image, clone it for output, and then convert it to grayscale

    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    gray = cv2.GaussianBlur(gray, (5, 5), 0);
    gray = cv2.medianBlur(gray, 5)

    # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)

    # gray = cv2.Canny(gray, 90, 120)

    kernel = np.ones((3, 3), np.uint8)

    gray = cv2.erode(gray, kernel, iterations=1)
    # gray = erosion
    #
    # gray = cv2.dilate(gray, kernel, iterations=1)
    # gray = dilation

    # get the size of the final image
    # img_size = gray.shape
    # print img_size
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=0.9, minDist=100, param1=30, param2=65, minRadius=5,
                               maxRadius=0)
    # print circles
    no_h = 0
    no_q = 0
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle in the image

            if r < 55:
                objType = 'quarter'
                color = (0, 255, 0)
                no_q += 1
            else:
                objType = 'half'
                color = (0, 0, 255)
                no_h += 1
            cv2.putText(output, objType, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            cv2.circle(output, (x, y), r, color, 4)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # time_end = time.time()
        # fps = 1 / (time_end - time_start)
        # fps = f'CPU: Frame rate: {str(int(fps))}'
        # cv2.putText(gray, fps, (50, 50), font, 1, (0, 0, 0), 1)
        # time_start = time_end
        # Display the resulting frame
        cv2.imshow('gray', gray)
    # cv2.putText(output, objType, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
    time_end = time.time()
    fps = 1 / (time_end - time_start)
    fps = f'CPU: Frame rate: {str(int(fps))}, NoOfQuarter: {no_q}, NoOfHalf:{no_h}'
    cv2.putText(output, fps, (10, 30), font, 0.5, (0, 0, 0), 1)
    time_start = time_end
    cv2.imshow('frame', output)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
