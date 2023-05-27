import time
import cv2
import sys
import numpy as np

__autor__ = 'Sanjib Sarkar'
__email__ = 'sanjib.sarkar@acquiredata.com'
__status__ = 'Prototype'

sys.path.append("../files")


class ProcessFilter:
    """ Image filter OpenCV"""
    def __init__(self, window_title_name, camOjt, scale=0.6):
        self.camOjt = camOjt
        self.wt = window_title_name
        self.width = 1080
        self.height = 800
        self.scale = scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position = (50, 50)
        self.fontScale = 1
        self.fontColor = (255, 0, 255)
        self.thickness = 4

    def bilateral_gpu(self):
        'Create frame on GPU'
        gpu_frame = cv2.cuda_GpuMat()
        time_start = 0
        while True:
            read_ok_g, frame_g = self.camOjt.read()
            if not read_ok_g:
                print(f'No frame- GPU')
                break
            ' upload the frame to the GPU'
            gpu_frame.upload(frame_g)
            frame_resized = cv2.cuda.resize(gpu_frame, (int(self.width * self.scale), int(self.height * self.scale)))
            frame_gray = cv2.cuda.bilateralFilter(frame_resized, 30, 32, 32)
            dnld = frame_gray.download()
            time_end = time.time()
            fps = 1 / (time_end - time_start)
            fps = f'GPU: Frame rate: {str(int(fps))}'
            cv2.putText(dnld, fps, self.position, self.font, self.fontScale, self.fontColor, self.thickness)
            # yield (self.wt, dnld)
            cv2.imshow(self.wt, dnld)

            # Close video window by pressing 'x'
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
            time_start = time_end
        # self.camOjt.release()
        # cv2.destroyAllWindows()

    def bilateral_cpu(self):
        time_start = 0
        while True:
            read_ok, imgc = self.camOjt.read()
            resized = cv2.resize(imgc, (int(self.width * self.scale), int(self.height * self.scale)))
            filter = cv2.bilateralFilter(resized, 30, 32, 32)

            time_end = time.time()
            fps = 1 / (time_end - time_start)
            fps = f'CPU: Frame rate: {str(int(fps))}'

            cv2.putText(filter, fps, self.position, self.font, self.fontScale, self.fontColor, self.thickness)
            cv2.imshow(self.wt, filter)

            # Close video window by pressing 'x'
            if cv2.waitKey(1) & 0xFF == ord('x'):
                print(f'No frame- CPU')
                break
            time_start = time_end

        # self.camOjt.release()
        # cv2.destroyAllWindows()


class ObjectDetection:
    def __init__(self, window_title_name, cam_obj, scale=0.80):
        self.camOjt = cam_obj
        self.wt = window_title_name
        self.width = 1080
        self.height = 800
        self.scale = scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.position = (50, 50)
        self.fontScale = 1
        self.fontColor = (255, 0, 255)
        self.thickness = 4
        # net = cv2.dnn.readNet(r"C:\Users\Public\Documents\opencv-project\files\yolov31.cfg",
        #                       r"C:\Users\Public\Documents\opencv-project\files\yolov31.weights")

    def obj_det_gpu(self):
        net = cv2.dnn.readNet(r"C:\Users\Public\Documents\opencv-project\files\yolov31.cfg",
                              r"C:\Users\Public\Documents\opencv-project\files\yolov31.weights")

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        classes = []
        with open("files\coco1.names", "r") as f:
            classes = f.read().splitlines()
            # print(classes)

        # cap = cv2.VideoCapture(1)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(100, 3))
        time_start = 0
        while True:
            # print(f'GPU Detection')
            _, img = self.camOjt.read()
            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

            time_end = time.time()
            fps = 1 / (time_end - time_start)
            fps = f'GPU: Frame rate: {str(int(fps))}'
            # print(fps)
            cv2.putText(img, fps, self.position, self.font, self.fontScale, self.fontColor, self.thickness)
            cv2.imshow(self.wt, img)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
            time_start = time_end

    def obj_det_cpu(self):
        net = cv2.dnn.readNet(r"C:\Users\Public\Documents\opencv-project\files\yolov3.cfg",
                              r"C:\Users\Public\Documents\opencv-project\files\yolov3.weights")

        classes = []
        with open("files\coco.names", "r") as f:
            classes = f.read().splitlines()
            # print(classes)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(100, 3))
        time_start = 0
        while True:
            _, img = self.camOjt.read()
            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

            time_end = time.time()
            fps = 1 / (time_end - time_start)
            fps = f'GPU: Frame rate: {str(int(fps))}'
            cv2.putText(img, fps, self.position, self.font, self.fontScale, self.fontColor, self.thickness)
            cv2.imshow(self.wt, img)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
            time_start = time_end


if __name__ == '__main__':
    pass
    # net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')
