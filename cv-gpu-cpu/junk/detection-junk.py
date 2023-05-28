# import the necessary packages
import cv2
import numpy as np
import time

# sys.path.append("../files")

position = (25, 25)
fontScale = 2
fontColor = (255, 0, 250)
thickness = 2

# net = cv2.dnn.readNet(r"C:\Users\Public\Documents\opencv-project\files\yolov3.cfg",
#                       r"C:\Users\Public\Documents\opencv-project\files\yolov3.weights")
net = cv2.dnn.readNet(r"C:\Users\Public\Documents\opencv\cv-gpu-cpu\files\yolov3.cfg",
                      r"C:\Users\Public\Documents\opencv\cv-gpu-cpu\files\yolov3.weights")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open(r"C:\Users\Public\Documents\opencv\cv-gpu-cpu\files\coco.names", "r") as f:
    classes = f.read().splitlines()
    # print(classes)

cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
time_start = 0
while True:
    _, img = cap.read()
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
    cv2.putText(img, fps, position, font, fontScale, fontColor, thickness)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    time_start = time_end

cap.release()
cv2.destroyAllWindows()
