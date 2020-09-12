from torch.utils.data import Dataset
from PIL import Image
import os
import pickle

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


class CroppedDataset(Dataset):
    def __init__(self, data_path, task_type=None, model_type=None):
    	with open(data_path, 'rb') as f:
        	self.frames = pickle.load(f)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


