import numpy as np
import pickle
import pylab
from BackgroundPredictor import BackgroundPredictor


class BackgroundAnnotator(object):
    """
    Annotator
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def annotate(self, frame):
        """ annotate object """
        # get background
        background = np.zeros_like(frame)
        foreground = np.zeros_like(frame)
        weights = []
        weights = self.predictor.predict(frame)

        # TODO make it binary with threshold

        # subtract background & foreground
        for c in range(3):
            background[:, :, c] += weights * (frame[:, :, c])
            foreground[:, :, c] += (1 - weights) * (frame[:, :, c])

        return background, foreground


def load_predictor():
    with open('../data/background.pkl', 'r') as f:
        data = pickle.load(f)

    predictor = BackgroundPredictor(data)
    return predictor


def show_image(image):
    fig = pylab.figure()
    fig.suptitle('display image', fontsize=20)
    pylab.imshow(image.astype(np.uint8))


def load_frames():
    with open('../data/test.pkl', 'r') as f:
        data = pickle.load(f)
    # print(data)

    # load frames as float
    frames = np.array(data, dtype=np.int8)
    frames = np.mod(frames + 256, 256)
    frames = frames.astype(np.float32)
    return frames


if __name__ == '__main__':
    predictor = load_predictor()
    frames = load_frames()

    # TODO annotate objects in frame list
    annotator = BackgroundAnnotator(predictor)
    for frame in frames:
        [background, foreground] = annotator.annotate(frame)
        show_image(foreground)
    pylab.show()
