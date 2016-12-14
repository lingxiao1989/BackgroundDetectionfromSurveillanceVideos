import numpy as np
import pickle
import pylab
import imageio
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

        # make it binary with threshold
        weights[weights >= 0.8] = 1
        weights[weights < 0.8] = 0

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
    #frames = load_frames()
    annotator = BackgroundAnnotator(predictor)
    reader = imageio.get_reader('../data/CCTV Full HD 1080p 10Fps Zoom 15X.mp4')
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer('../data/test2_foreground.mp4', fps=fps)

    # annotate objects in frame list
    for frame in reader:
        #show_image(frame)
        [background,foreground]=annotator.annotate(frame.astype(np.float32))
        writer.append_data(np.hstack((frame,background)))
    writer.close()
