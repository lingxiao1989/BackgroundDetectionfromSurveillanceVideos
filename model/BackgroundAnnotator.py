import numpy as np


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
