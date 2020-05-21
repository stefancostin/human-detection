import cv2

class TrackerFactory:

    OPENCV_OBJECT_TRACKERS = {
        "CSRT": cv2.TrackerCSRT_create,
        "KCF": cv2.TrackerKCF_create,
        "BOOSTING": cv2.TrackerBoosting_create,
        "MIL": cv2.TrackerMIL_create,
        "TLD": cv2.TrackerTLD_create,
        "MEDIANFLOW": cv2.TrackerMedianFlow_create,
        "MOSSE": cv2.TrackerMOSSE_create
    }

    def __init__(self, tracker):
        self.tracker = tracker

    def get_instance(self):
        return self.OPENCV_OBJECT_TRACKERS[self.tracker]()