import numpy as np
import pickle
import random
 

from PIL import Image

def generate_trace(size, min_size=512 ):
    step=[]
    for i in range (0,(size[1]-min_size),50):
        y_start=i
        y_end=i+min_size       
        for j in range (0,(size[0]-min_size),50):
            crop=[]
            x_start=j
            x_end=j+min_size
            crop=[x_start, y_start, x_end, y_end]
            step.append(crop)
    return step

def generate_GTP_data(img, min_size=1024 ):
    weight=img.size[0]
    height=img.size[1]
    overlap_ratio=random.randint(1,99)/100
    step=[]
    for i in range (0,(height-min_size),50):
        y_start=i
        for j in range (0,(weight-min_size),50):
            overlap_ratio=int(min_size*random.randint(1,99)/100)
            crop1=[]
            crop2=[]
            crop3=[]
            x_start=j
            crop1=[x_start, y_start, x_start+min_size, y_start+min_size]
            crop2=[x_start+overlap_ratio, y_start+overlap_ratio, x_start+overlap_ratio+min_size, y_start+overlap_ratio+min_size]
            crop3=[x_start, y_start, x_start+overlap_ratio+min_size, y_start+overlap_ratio+min_size]
            step.append(crop1)
            step.append(crop2)
            step.append(crop3)
    return step

'''
def main():
    # load frames as list
    #frames = load_frames()

    img = Image.open("../data/surveillance_pic1.jpg")
    
    trace_point = generate_trace(img.size)
    frames = []

    for roi in trace_point:
        print(roi)
        cropped = img.crop(roi)
        frames.append(cropped)

    #cropped.show()
    #print('type:{}, shape:{}, range:[{}, {}]'.format(frames.dtype, frames.shape, np.min(frames), np.max(frames)))
    #show_image(frames[0])
    #print (len(frames))

    with open('../data/test.pkl', 'wb') as f:
        pickle.dump(frames, f)
'''    
def main():
    # load frames as list
    #frames = load_frames()

    img = Image.open("../data/surveillance_pic1.jpg")
    
    
    sample = generate_GTP_data(img)
    frames = []

    for roi in trace_point:
        print(roi)
        cropped = img.crop(roi)
        frames.append(cropped)

    #cropped.show()
    #print('type:{}, shape:{}, range:[{}, {}]'.format(frames.dtype, frames.shape, np.min(frames), np.max(frames)))
    #show_image(frames[0])
    #print (len(frames))

    with open('../data/test.pkl', 'wb') as f:
        pickle.dump(frames, f)

if __name__ == '__main__':
    main()
