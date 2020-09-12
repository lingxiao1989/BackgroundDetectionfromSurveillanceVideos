import pickle

from PIL import Image
from utils.dataset import CroppedDataset
import torch

def main():
    """ dataset preparation """
    file_path = 'data/test.pkl'
    train_set = CroppedDataset(file_path)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=16,
        shuffle=False, num_workers=int(4),
        #collate_fn=BaiduCollate(opt.imgH, opt.imgW, keep_ratio=False)
    )

    # load frames as list
    #frames = load_frames()
    #print('type:{}, shape:{}, range:[{}, {}]'.format(frames.dtype, frames.shape, np.min(frames), np.max(frames)))
    #frames[0].show()

if __name__ == '__main__':
    main()
