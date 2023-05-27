import cv2

ap = cv2.VideoCapture(r'C:\Users\Public\Documents\opencv-project\files\chaplin.mp4')

# # import libraries
# from vidgear.gears import CamGear
# import cv2
#
# stream = CamGear(source='https://youtu.be/dQw4w9WgXcQ', stream_mode=True,
#                  logging=True).start()  # YouTube Video URL as input
#
# # infinite loop
# while True:
#
#     frame = stream.read()
#     # read frames
#
#     # check if frame is None
#     if frame is None:
#         # if True break the infinite loop
#         break
#
#     # do something with frame here
#
#     cv2.imshow("Output Frame", frame)
#     # Show output window
#
#     key = cv2.waitKey(1) & 0xFF
#     # check for 'q' key-press
#     if key == ord("q"):
#         # if 'q' key-pressed break out
#         break
#
# cv2.destroyAllWindows()
# # close output window
#
# # safely close video stream.
# stream.stop()


# import cv2
#
# uri = "https://www.youtube.com/watch?v=_LcMwmdAPiM"
#
# gst_pipeline = "uridecodebin uri=%s ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1" % (
#     uri)
# cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
#
# if not cap.isOpened():
#     print('Failed to open source')
#     exit(-1)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print('Failed to read from source')
#         break
#
#     cv2.imshow('Test URI', frame)
#     cv2.waitKey(1)
#
# cap.release()
# #
# # url = "https://www.youtube.com/watch?v=_LcMwmdAPiM"
# video = pafy.new(url)
# best = video.getbest(preftype="webm")
# # documentation: https://pypi.org/project/pafy/
#
# capture = cv2.VideoCapture(best.url)
# while True:
#     check, frame = capture.read()
#     print(check, frame)
#
#     cv2.imshow('frame', frame)
#     cv2.waitKey(10)
#
#     capture.release()
#     cv2.destroyAllWindows()
