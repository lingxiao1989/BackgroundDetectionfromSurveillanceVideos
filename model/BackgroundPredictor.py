import numpy as np


class BackgroundPredictor(object):
    """
    Predictor
    """

    def __init__(self, background):
        self.background = background
        self.grey = self.to_grey()

    def predict(self, frame):
        # assure equal size
        np.testing.assert_array_equal(frame.shape, self.background.shape)

        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        frame_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        sigma = 25
        prob = np.exp(- ((frame_gray - self.grey)/sigma)**2)

        # TODO smooth probability by (y,x)

        return prob

    def optimize(self, background):
        self.background = background
        self.grey = self.to_grey()

    def to_grey(self):
        frame = self.background
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grey
