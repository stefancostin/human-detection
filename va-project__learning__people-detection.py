# import the necessary packages
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('run.mp4')

# the output will be written to output.avi
out = cv2.VideoWriter(
    'altceva.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)





    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    # (boxes, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
    #     padding=(8, 8), scale=1.05)
    # (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
    #     padding=(8, 8), scale=1.05)


    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    pick = non_max_suppression(boxes, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)





    # # detect people in the image
    # (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
    #     padding=(8, 8), scale=1.05)
    #
    # # apply non-maxima suppression to the bounding boxes using a
    # # fairly large overlap threshold to try to maintain overlapping
    # # boxes that are still people
    # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    # pick = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
    #
    # # draw the final bounding boxes
    # for (xA, yA, xB, yB) in pick:
    #     cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)



    # Write the output video
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()

# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)