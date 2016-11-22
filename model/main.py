import cv2,numpy as np
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
    # load the video
    vidcap = cv2.VideoCapture('Surveillance Feed - Parking Lot.mp4')
    count = 0
    
    # load frames as list
    frames = []
    
    # take one gray frame in each loop and save the frame into the list
    while (count<3):
        vidcap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,count*1000) 
        success,image = vidcap.read()
        print 'Read a new frame: ', success
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        count += 1



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
