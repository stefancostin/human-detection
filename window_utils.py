import cv2

class WindowUtils:

    # resize a window to a constant width or height provided as input
    # while maintaining the original aspect ratio of the video frame
    def resize(self, frame, width=None, height=None):
        aspect_ratio = float(frame.shape[1]) / float(frame.shape[0])

        if width is None and height is None:
            print('No height or width has been provided for resizing')
            return frame

        elif width is not None and height is None:
            height = width / aspect_ratio

        elif width is not None and height is not None:
            width = height / aspect_ratio

        dim = (int(width), int(height))
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def get_height(self, frame):
        return frame.shape[0]

    def get_width(self, frame):
        return frame.shape[1]
