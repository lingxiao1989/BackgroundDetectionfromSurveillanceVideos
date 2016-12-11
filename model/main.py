import numpy as np
import pickle
import BackgroundPredictor
import BackgroundAnnotator


# Expectation: output estimated background
def e_step(predictor, frames):
    # foreach list frames
    #   predict pixel level likelihood as background
    for frame in frames:
        predictor.predict(frame)

    # foreach pixel
    #   calculate weighted average for background

    # return background
    return np.zeros([1, 1])


# Maximization: output better predictor
def m_step(background, predictor):
    # given estimated background
    # optimize predictor errors
    predictor.optimize(background)

    return predictor


def main():
    # load frames as list
    with open('../data/test.pkl', 'r') as f:
        data = pickle.load(f)
    print(data)
    frames = []

    # init background
    background = np.zeros([1, 1])

    # iterate e_step and m_step
    # output: background + predictor
    predictor = BackgroundPredictor.BackgroundPredictor(background)
    while True:
        background = e_step(predictor, frames)
        predictor = m_step(background, predictor)
        if True:
            break

    # annotate objects in frame list
    annotator = BackgroundAnnotator.BackgroundAnnotator(predictor)
    for frame in frames:
        annotator.annotate(frame)

    # output


if __name__ == '__main__':
    main()
