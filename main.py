from __future__ import absolute_import
import cv2
from math import sqrt
from person import Person
from person_tracker import PersonTracker
from tracker_factory import TrackerFactory
from window_utils import WindowUtils

# install opencv-contrib-python


# initialize colors
# green is the color of detection done frame by frame
# blue is the color of the already tracked persons
green = (255, 0, 0)
blue = (0, 0, 255)

# initialize the video stream
vs = cv2.VideoCapture('videos/run.mp4')

# initialize the HOG descriptor / person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# euclidean distance value for knowing when
# the detection runs on already tracked persons
similarity_threshold = 80.0

# initialize tracker factory
tracker_type = 'KCF'
tf = TrackerFactory(tracker_type)

# initialize object that holds information on the people being tracked
person_tracker = PersonTracker()

# initialize utility object that helps in manipulating the window frame
window_utils = WindowUtils()

# initialize frames per second variable
fps = None

while True:
    # read a new frame
    ok, frame = vs.read()
    if not ok:
        break

    # resize the video frame
    frame = window_utils.resize(frame, width=640)
    height = window_utils.get_height(frame)

    # start timer
    timer = cv2.getTickCount()

    # calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # detect people in the image
    (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # draw boxes around the people that have been detected
    for bbox in boxes:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 1, 1)

    # track the people that have been previously detected
    for i in range(len(boxes)):
        # save coordinates as tuple
        bbox = tuple(i for i in boxes[i])

        is_same_person = False
        # discard duplicate trackers using euclidean distance
        for person in person_tracker.list():
            distance = sqrt(sum([(a - b) ** 2 for a, b in zip(person.bbox, bbox)]))
            if distance < similarity_threshold:
                is_same_person = True
                break

        if is_same_person == True:
            continue

        # create new trackable object for each person identified
        tag = 'person_' + str(person_tracker.total_count())
        person = Person(bbox, tag, tf.get_instance())

        # start tracking
        person.track(frame)

        # add trackable object to the list of tracked persons
        person_tracker.add(person)

    # append tracked persons that have exited the frame
    tracking_failed = []

    # update tracker
    for person in person_tracker.list():
        ok = person.update(frame)

        if ok:
            # draw bounding boxes on tracking success
            bbox = person.bbox
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, green, 1, 1)
        else:
            # remove tracked persons on tracking failure
            tracking_failed.append(person)
            cv2.putText(frame, "Tracking failure detected", \
                        (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2)

    # remove the persons on which tracking has stopped
    person_tracker.remove(tracking_failed)

    # display number of people being tracked
    cv2.putText(frame, "Currently tracked : " + str(person_tracker.size()), \
                (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)

    # display total number of people tracked
    cv2.putText(frame, "Total tracked : " + str(person_tracker.total_count()), \
                (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 170, 50), 2)

    # display tracker type on frame
    cv2.putText(frame, str(tracker_type) + " Tracker", \
                (80, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 1)

    # display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), \
                (80, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 1)

    # display result
    cv2.imshow("Tracking", frame)

    # press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break