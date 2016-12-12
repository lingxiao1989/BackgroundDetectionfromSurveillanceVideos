import numpy as np
import pickle
from BackgroundPredictor import BackgroundPredictor


class BackgroundAnnotator(object):
    """
    Annotator
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def annotate(self, frame):
        """ annotate object """
        # predict background
        self.predictor.predict(frame)

        # subtract background

        return np.zeros_like(frame)


def load_predictor():
    with open('../data/background.pkl', 'r') as f:
        data = pickle.load(f)

    predictor = BackgroundPredictor(data)
    return predictor


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
        annotator.annotate(frame)
