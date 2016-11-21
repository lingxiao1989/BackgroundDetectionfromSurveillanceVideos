import numpy as np
import BackgroundPredictor as Predictor
import BackgroundAnnotator as Annotator


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
    frames = []

    # init background
    background = np.zeros([1, 1])

    # iterate e_step and m_step
    # output: background + predictor
    predictor = Predictor(background)
    while True:
        background = e_step(predictor, frames)
        predictor = m_step(background, predictor)
        if True:
            break

    # annotate objects in frame list
    annotator = Annotator(predictor)
    for frame in frames:
        annotator.annotate(frame)

    # output


if __name__ == '__main__':
    main()
