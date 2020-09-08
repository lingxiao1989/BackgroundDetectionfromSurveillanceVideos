import pickle

from PIL import Image

def load_frames():
    file_path = '../data/test.pkl'
    with open(file_path, 'rb') as f:
        frames = pickle.load(f)
    # print(data)

    # load frames as float
    #frames = np.array(data, dtype=np.int8)
    #frames = np.mod(frames + 256, 256)
    #frames = frames.astype(np.float32)
    return frames


def main():
    # load frames as list
    frames = load_frames()
    #print('type:{}, shape:{}, range:[{}, {}]'.format(frames.dtype, frames.shape, np.min(frames), np.max(frames)))
    frames[0].show()

if __name__ == '__main__':
    main()