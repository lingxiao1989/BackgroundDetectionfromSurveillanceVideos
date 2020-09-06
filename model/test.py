import numpy as np
import pickle

from PIL import Image

def load_frames():
    file_path = '../data/background.pkl'
    file_path = str.encode(file_path)
    data = pickle.load(open(file_path))
    #with open(file_path, 'r') as f:
        #data = pickle.load(f)
    # print(data)

    # load frames as float
    frames = np.array(data, dtype=np.int8)
    frames = np.mod(frames + 256, 256)
    frames = frames.astype(np.float32)
    return frames

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

def main():
    # load frames as list
    #frames = load_frames()

    img = Image.open("../data/surveillance_pic1.jpg")
    
    trace_point = generate_trace(img.size)
    print(img.size)
    print(trace_point)

    #cropped = img.crop((0, 0, 512, 512))

    #cropped.show()

    #print('type:{}, shape:{}, range:[{}, {}]'.format(frames.dtype, frames.shape, np.min(frames), np.max(frames)))
    #show_image(frames[0])
    #print (len(frames))


    
if __name__ == '__main__':
    main()
