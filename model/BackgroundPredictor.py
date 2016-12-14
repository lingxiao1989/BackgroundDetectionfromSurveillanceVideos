import numpy as np


class BackgroundPredictor(object):
    """
    Predictor
    """

    def __init__(self, background):
        self.background = background
        self.grey = self.to_grey()
        self.sigma = 25  # to decrease

    def predict(self, frame):
        # assure equal size
        np.testing.assert_array_equal(frame.shape, self.background.shape)

        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        frame_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        prob = np.exp(- ((frame_gray - self.grey)/self.sigma)**2)

        # smooth probability by (y,x)
        prob = self.gaussian_filter(prob)

        return prob

    def optimize(self, background):
        self.background = background
        self.grey = self.to_grey()

    def to_grey(self):
        frame = self.background
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grey

    @staticmethod
    def gaussian_filter(prob):
        """ 3x3 """
        gaussian_grid = [[1.0/16, 1.0/8, 1.0/16], [1.0/8, 1.0/4, 1.0/8], [1.0/16, 1.0/8, 1.0/16]]

        output = prob * gaussian_grid[1][1]

        xmax = prob.shape[1]-1
        for y in range(1, prob.shape[0]-1):
            output[y-1, :] += prob[y, :] * gaussian_grid[1][2]

            output[y-1, 0:(xmax-1)] += prob[y, 1:xmax] * gaussian_grid[0][2]
            output[y-1, xmax] += prob[y, xmax] * gaussian_grid[0][2]

            output[y-1, 1:xmax] += prob[y, 0:(xmax-1)] * gaussian_grid[2][2]
            output[y-1, 0] += prob[y, 0] * gaussian_grid[2][2]

            output[y, 0:(xmax - 1)] += prob[y, 1:xmax] * gaussian_grid[0][1]
            output[y, xmax] += prob[y, xmax] * gaussian_grid[0][1]

            output[y, 1:xmax] += prob[y, 0:(xmax - 1)] * gaussian_grid[2][1]
            output[y, 0] += prob[y, 0] * gaussian_grid[2][1]

            output[y + 1, :] += prob[y, :] * gaussian_grid[1][0]

            output[y + 1, 0:(xmax - 1)] += prob[y, 1:xmax] * gaussian_grid[0][0]
            output[y + 1, xmax] += prob[y, xmax] * gaussian_grid[0][0]

            output[y + 1, 1:xmax] += prob[y, 0:(xmax - 1)] * gaussian_grid[2][0]
            output[y + 1, 0] += prob[y, 0] * gaussian_grid[2][0]

        for y in [0, prob.shape[0]-1]:
            output[y, 0:(xmax - 1)] += prob[y, 1:xmax] * (gaussian_grid[0][1] + gaussian_grid[0][2])
            output[y, xmax] += prob[y, xmax] * (gaussian_grid[0][1] + gaussian_grid[0][2])

            output[y, 1:xmax] += prob[y, 0:(xmax - 1)] * (gaussian_grid[2][1] + gaussian_grid[2][2])
            output[y, 0] += prob[y, 0] * (gaussian_grid[2][1] + gaussian_grid[2][2])

            output[y, :] += prob[y, :] * gaussian_grid[1][2]

        return output
