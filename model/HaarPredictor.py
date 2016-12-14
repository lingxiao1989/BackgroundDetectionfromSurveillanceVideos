import numpy as np
import pickle
from BackgroundPredictor import BackgroundPredictor
import pylab
import imageio
import cv2

class HaarPredictor():
    """
    predictor
    """

    def __init__(self):
		#self.predictor = predictor
		self.car_cascade=cv2.CascadeClassifier('../data/cars3.xml')
		
		
    def predict(self, frame):
        """ predict object """
        # predictor cars

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray)
        classifier = np.ones_like(gray)
        for (x,y,w,h) in cars:
            if w>100:
                classifier[x:x+w,y:y+h]=0.1
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #gray_crop=frame[y:y+h, x:x+w]
        #output='../data/cars.jpg'
        #cv2.imwrite(output,frame)    
        #self.weights.append(classifier)
        return classifier
		
    def demo(self, frame):
        """ predict object """
        # predictor cars

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray)
        classifier = np.ones_like(gray)
        for (x,y,w,h) in cars:
            if w>100:
                #classifier[x:x+w,y:y+h]=0.1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #gray_crop=frame[y:y+h, x:x+w]
        #output='../data/cars.jpg'
        #cv2.imwrite(output,frame)    
        #self.weights.append(classifier)
        return frame
	
		
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
    #frames = frames.astype(np.float32)
    frames = frames.astype(np.uint8)
    return frames

if __name__ == '__main__':
    predictor=HaarPredictor()
    weights=[]
    frames = load_frames()	
    for frame in frames:
        show_image(predictor.demo(frame))
    pylab.show()