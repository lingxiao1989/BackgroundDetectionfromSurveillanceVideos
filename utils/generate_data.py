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
    width=img.size[0]
    height=img.size[1]
    step=[]
    for i in range (0,(height-2*min_size),50):
        y_start=i
        for j in range (0,(width-2*min_size),50):
            overlap_pixels=int(min_size*random.randint(1,99)/100)
            temp=[]
            x_start=j
            crop1=[x_start, y_start, x_start+min_size, y_start+min_size]
            crop2=[x_start+overlap_pixels, y_start+overlap_pixels, x_start+overlap_pixels+min_size, y_start+overlap_pixels+min_size]
            crop3=[x_start, y_start, x_start+overlap_pixels+min_size, y_start+overlap_pixels+min_size]
            temp.append(crop1)
            temp.append(crop2)
            temp.append(crop3)
            step.append(temp)

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
    
    IMAGE_SIZE=512
    sample = generate_GTP_data(img, IMAGE_SIZE)
    frames = []
    i=0

    for rois in sample:
        print(rois)
        x1=rois[0]
        x2=rois[1]
        y=rois[2]
        imgs=[]
        imgs.append(img.crop(x1))
        imgs.append(img.crop(x2))
        imgs.append(img.crop(y).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS))
        #img.crop(x1).show()
        i+=1
        frames.append(imgs)
        if i>150:
            break

    #cropped.show()
    #print('type:{}, shape:{}, range:[{}, {}]'.format(frames.dtype, frames.shape, np.min(frames), np.max(frames)))
    #show_image(frames[0])
    #print (len(frames))

    with open('../data/test.pkl', 'wb') as f:
        pickle.dump(frames, f)

if __name__ == '__main__':
    main()
