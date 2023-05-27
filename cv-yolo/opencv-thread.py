import cv2
import threading


class Cam(threading.Thread):
    def __init__(self, name, camID):
        threading.Thread.__init__(self)
        self.name = name
        self.camID = camID
        self.cf = CaptureFrame(self.name, camID=camID)

    def run(self):
        self.cf.capture()


class CaptureFrame:
    def __init__(self, name, camID):
        self.name = name
        self.camID = camID

    def capture(self):
        # cv2.nameWindow(self.name)
        cam = cv2.VideoCapture(self.camID)
        if cam.isOpened():
            status, frame = cam.read()
        else:
            status = False
        while status:
            cv2.imshow(self.name, frame)
            status, frame = cam.read()
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
        cv2.destroyWindow(self.name)


if __name__ == '__main__':
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    Cam('cam1', 0).start()
    Cam('cam2', 0).start()
