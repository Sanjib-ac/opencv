import time
import cv2
from matplotlib import pyplot as plt
import threading
import tkinter as tk
import pafy  # pafy allows us to read videos from youtube.
import SupportingClasses as Sc

__autor__ = 'Sanjib Sarkar'
__email__ = 'sanjib.sarkar@acquiredata.com'
__status__ = 'Prototype'


class Cam(threading.Thread):
    def __init__(self, cam_object, do='bi', platform='GPU'):
        threading.Thread.__init__(self, daemon=True)
        self.wt = platform
        self.cam = cam_object
        self.filter = Sc.ProcessFilter(self.wt, self.cam)  # use a filter
        self.obj_det = Sc.ObjectDetection(self.wt, self.cam)
        self.ptfrm = platform
        self.do = do
        self.start()

    def run(self):
        match self.do:
            case 'bi':
                if self.ptfrm == 'GPU':
                    self.filter.bilateral_gpu()
                elif self.ptfrm == 'CPU':
                    self.filter.bilateral_cpu()
            case 'detection':
                if self.ptfrm == 'GPU':
                    print(f'GPU detection started')
                    self.obj_det.obj_det_gpu()
                elif self.ptfrm == 'CPU':
                    print(f'CPU detection started')
                    self.obj_det.obj_det_cpu()
                    # print('not yet implemented CPU.')
            case _:
                print('No match')


if __name__ == '__main__':
    root = tk.Tk()
    print(f'OpenCV version: {cv2.__version__},: DeviceCount: {cv2.cuda.getCudaEnabledDeviceCount()}')

    # URL = "https://www.youtube.com/watch?v=1LCb1PVqzeY"  # URL to parse
    # play = pafy.new(URL)
    # best = play.getbest(preftype="mp4")
    # cap = cv2.VideoCapture(best.url)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(r"C:\Users\Public\Documents\opencv-project\files\Chase.mp4")
    # width = 1080
    # height = 800
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    c1 = Cam(cam_object=cap, do='detection', platform='GPU')
    c2 = Cam(cam_object=cap, do='detection', platform='CPU')
    #
    c1.join()
    c2.join()
