import numpy as np
import pickle
import pylab
from BackgroundPredictor import BackgroundPredictor


# Expectation: output estimated background
def e_step(predictor, frames):
    weights = []
    # foreach list frames
    #   predict pixel level likelihood as background
    for frame in frames:
        # print(frame.shape)
        weights.append(predictor.predict(frame))

    background = np.zeros_like(frames[0])

    # foreach pixel
    #   calculate weighted average for background
    for i in range(len(weights)):
        for c in range(3):
            background[:, :, c] += weights[i] * (frames[i, :, :, c] - predictor.background[:, :, c])

    for c in range(3):
        background[:, :, c] /= np.sum(weights, axis=0)

    # return background
    return background + predictor.background


# Maximization: output better predictor
def m_step(background, predictor):
    # given estimated background
    # optimize predictor errors
    predictor.optimize(background)

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


def main():
    # load frames as list
    frames = load_frames()
    print('type:{}, shape:{}, range:[{}, {}]'.format(frames.dtype, frames.shape, np.min(frames), np.max(frames)))
    show_image(frames[0])

    # init background
    background = np.sum(frames, axis=0) / float(frames.shape[0])
    print('range of background:[{}, {}]'.format(np.min(background), np.max(background)))
    show_image(background)

    # TODO better initial background with external predictor

    # iterate e_step and m_step
    # output: background + predictor
    predictor = BackgroundPredictor(background)
    iter = 0
    while True:
        background = e_step(predictor, frames)
        predictor = m_step(background, predictor)

        # damp sigma
        predictor.sigma *= 0.95

        if iter > 15:
            break
        if iter % 5 == 0:
            print('iter: {}, range:[{}, {}]'.format(iter, np.min(background), np.max(background)))
            show_image(background)
        iter += 1

    # output
    dump_data(background.astype(np.uint8), 'background.pkl')
    show_image(background)


def show_image(image):
    fig = pylab.figure()
    fig.suptitle('display image', fontsize=20)
    pylab.imshow(image.astype(np.uint8))

    # pylab.show()


def dump_data(obj, filename):
    with open('../data/' + filename, 'w') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    main()
    pylab.show()
