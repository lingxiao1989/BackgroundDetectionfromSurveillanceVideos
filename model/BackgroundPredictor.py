import numpy as np


class BackgroundPredictor(object):
    """
    Predictor
    """

    def __init__(self, background):
        self.background = background

    def predict(self, frame):
        # assure equal size
        np.testing.assert_array_equal(frame.shape, self.background.shape)

        return np.zeros_like(frame)

    def optimize(self, background):
        pass
