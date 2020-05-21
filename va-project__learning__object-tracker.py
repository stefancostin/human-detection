from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import numpy as np
import cv2

# install opencv-contrib-python

# OPENCV_OBJECT_TRACKERS = {
#     "csrt": cv2.TrackerCSRT_create,
#     "kcf": cv2.TrackerKCF_create,
#     "boosting": cv2.TrackerBoosting_create,
#     "mil": cv2.TrackerMIL_create,
#     "tld": cv2.TrackerTLD_create,
#     "medianflow": cv2.TrackerMedianFlow_create,
#     "mosse": cv2.TrackerMOSSE_create
# }
# tracker = OPENCV_OBJECT_TRACKERS["mil"]

tracker = cv2.TrackerKCF_create('KCF')

initBB = None
fps = None

# open webcam video stream
# cap = cv2.VideoCapture(0)
vs = cv2.VideoCapture('run.mp4')




# # loop over frames from the video stream
# while (True):
#     # Capture frame-by-frame
#     ret, frame = vs.read()
#
#     print(fps)
#     # resizing for faster detection
#     frame = cv2.resize(frame, (640, 480))
#
#     # check to see if we are currently tracking an object
#     if initBB is not None:
#         # grab the new bounding box coordinates of the object
#         (success, box) = tracker.update(frame)
#         # check to see if the tracking was a success
#         if success:
#             (x, y, w, h) = [int(v) for v in box]
#             cv2.rectangle(frame, (x, y), (x + w, y + h),
#                           (0, 255, 0), 2)
#         # update the FPS counter
#         fps.update()
#         fps.stop()
#         # initialize the set of information we'll be displaying on
#         # the frame
#         info = [
#             ("Tracker", tracker),
#             ("Success", "Yes" if success else "No"),
#             ("FPS", "{:.2f}".format(fps.fps())),
#         ]
#         # loop over the info tuples and draw them on our frame
#         for (i, (k, v)) in enumerate(info):
#             text = "{}: {}".format(k, v)
#             cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)




# Read first frame.
ok, frame = vs.read()

# Define an initial bounding box
bbox = (287, 23, 86, 320)

# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = vs.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, str(tracker) + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break