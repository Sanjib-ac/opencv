# Linked to git

import time
import cv2
from matplotlib import pyplot as plt
import threading
import tkinter as tk

__autor__ = 'Sanjib Sarkar'
__email__ = 'sanjib.sarkar@acquiredata.com'
__status__ = 'Prototype'

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = 1080
height = 800
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

font = cv2.FONT_HERSHEY_SIMPLEX
position = (50, 50)
fontScale = 1
fontColor = (255, 0, 255)
thickness = 4
# lineType = cv2.LINE_AA
scale = 0.60


def gpu_camera():
    'Create frame on GPU'
    gpu_frame = cv2.cuda_GpuMat()  # allocate memory in the GPU
    cny_detector = cv2.cuda.createCannyEdgeDetector(low_thresh=100, high_thresh=120)
    time_start = 0
    while True:
        read_ok_g, frame_g = cap.read()
        # cv2.imshow("Original", frame_g)
        if not read_ok_g:
            print(f'No frame')
            break
        ' upload the frame to the GPU'
        gpu_frame.upload(frame_g)

        frame_resized = cv2.cuda.resize(gpu_frame, (int(width * scale), int(height * scale)))
        # frame_gray = cv2.cuda.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        # cny_gframe = cny_detector.detect(frame_g)
        frame_gray = cv2.cuda.bilateralFilter(frame_resized, 30, 32, 32)

        dnld = frame_gray.download()

        time_end = time.time()
        fps = 1 / (time_end - time_start)
        fps = f'GPU: Frame rate: {str(int(fps))}'
        # cv2.putText(canny, fps, position, font, fontScale, fontColor,
        #             thickness, lineType)
        cv2.putText(dnld, fps, position, font, fontScale, fontColor, thickness)
        cv2.imshow("GPU", dnld)

        # Close video window by pressing 'x'
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
        time_start = time_end


def cpu_camera():
    # cap = webcam_object
    time_start = 0
    while True:
        read_ok, imgc = cap.read()
        if not read_ok:
            print(f'No frame')
            break
        resized = cv2.resize(imgc, (int(width * scale), int(height * scale)))
        filter = cv2.bilateralFilter(resized, 30, 32, 32)
        # cny = cv2.Canny(resz, 100, 120)

        time_end = time.time()
        fps = 1 / (time_end - time_start)
        fps = f'CPU: Frame rate: {str(int(fps))}'

        cv2.putText(filter, fps, position, font, fontScale, fontColor, thickness)
        cv2.imshow("CPU", filter)

        # Close video window by pressing 'x'
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
        time_start = time_end


if __name__ == '__main__':
    root = tk.Tk()
    threading.Thread(target=gpu_camera).start()

    threading.Thread(target=cpu_camera).start()

    # print(cv2.__version__)
    #
    # print(cv2.cuda.getCudaEnabledDeviceCount())
