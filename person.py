class Person:

    def __init__(self, bounding_box=None, identifier_tag=None, opencv_tracker=None):
        self.bbox = bounding_box
        self.tag = identifier_tag
        self.tracker = opencv_tracker

    def track(self, frame):
        self.tracker.init(frame, self.bbox)

    def update(self, frame):
        (ok, self.bbox) = self.tracker.update(frame)
        return ok