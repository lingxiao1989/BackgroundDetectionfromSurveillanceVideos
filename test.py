import pickle

from PIL import Image
from utils.dataset import CroppedDataset
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    
    return x

# pytorch helpfully makes it easy to download datasets, e.g. the common CIFAR-10 https://www.kaggle.com/c/cifar-10
#root = './'
#train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
#test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)


# run kmeans to get our codebook

def kmeans(x, ncluster, niter=10):
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # move each codebook element to be the mean of the pixels that assigned to it
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    return c

from torch.utils.data import Dataset

#TODO: modifiy ImageDataset for image stitching purpose
# input: each pt_dataset contains two overlapped cropped images and one origin ground truth image
# getitem: the flattened and cluster assigned sequence of two images and the flattened and cluster assigned sequence of ground truth image.
class ImageDataset(Dataset):
    """
    wrap up the pytorch CIFAR-10 dataset into our own, which will convert images into sequences of integers
    """
    
    def __init__(self, pt_dataset, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.perm = torch.arange(32*32) if perm is None else perm
        
        self.vocab_size = clusters.size(0)
        self.block_size = 32*32 - 1
        
    def __len__(self):
        return len(self.pt_dataset)
        
    def __getitem__(self, idx):
        x, index = self.pt_dataset[idx]
        x1 = torch.from_numpy(np.array(x[0])).view(-1, 3) # flatten out all pixels
        x2 = torch.from_numpy(np.array(x[1])).view(-1, 3) # flatten out all pixels
        X = torch.cat((x1,x2),0)
        X = X[self.perm].float() # reshuffle pixels with any fixed permutation and -> float
        X = ((X[:, None, :] - self.clusters[None, :, :])**2).sum(-1).argmin(1) # cluster assignments
        Y = torch.from_numpy(np.array(x[2])).view(-1, 3) # flatten out all pixels
        Y = Y[self.perm].float() # reshuffle pixels with any fixed permutation and -> float
        return X, Y # always just predict the next one in the sequence

from model.minGTP import GPT, GPTConfig, GPT1Config
'''
# we'll do something a bit smaller
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                  n_layer=12, n_head=8, n_embd=256)
model = GPT(mconf)

from model.trainer import Trainer, TrainerConfig

"""
Note that I am running on an 8-GPU V100 machine so each GPU has 32GB.
If you don't have as many computational resources you have to bring down
the batch_size until the model fits into your memory, and then you may
also need to adjust the learning rate (e.g. decrease it a bit). Alternatively,
you can use an even smaller model up above, bringing down the number of layers,
number of heads, and the embedding size.
"""

tokens_per_epoch = len(train_data) * train_dataset.block_size
train_epochs = 20 # todo run a bigger model and longer, this is tiny

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=train_epochs, batch_size=16*8, learning_rate=3e-3,
                      betas = (0.9, 0.95), weight_decay=0,
                      lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path='cifar10_model.pt',
                      num_workers=8)
trainer = Trainer(model, train_dataset, test_dataset, tconf)
trainer.train()

# load the state of the best model we've seen based on early stopping
checkpoint = torch.load('cifar10_model.pt')
model.load_state_dict(checkpoint)

# to sample we also have to technically "train" a separate model for the first token in the sequence
# we are going to do so below simply by calculating and normalizing the histogram of the first token
counts = torch.ones(ncluster) # start counts as 1 not zero, this is called "smoothing"
rp = torch.randperm(len(train_dataset))
nest = 5000 # how many images to use for the estimation
for i in range(nest):
    a, _ = train_dataset[int(rp[i])]
    t = a[0].item() # index of first token in the sequence
    counts[t] += 1
prob = counts/counts.sum()

%%time

from mingpt.utils import sample

n_samples = 32
start_pixel = np.random.choice(np.arange(C.size(0)), size=(n_samples, 1), replace=True, p=prob)
start_pixel = torch.from_numpy(start_pixel).to(trainer.device)
pixels = sample(model, start_pixel, 32*32-1, temperature=1.0, sample=True, top_k=100)

# for visualization we have to invert the permutation used to produce the pixels
iperm = torch.argsort(train_dataset.perm)

ncol = 8
nrow = n_samples // ncol
plt.figure(figsize=(16, 8))
for i in range(n_samples):
    pxi = pixels[i][iperm] # note: undo the encoding permutation
    
    plt.subplot(nrow, ncol, i+1)
    plt.imshow(C[pxi].view(32, 32, 3).numpy().astype(np.uint8))
    plt.axis('off')
'''
def main():
    set_seed(42)

    """ dataset preparation """
    file_path = 'data/test.pkl'
    train_data = CroppedDataset(file_path)
    '''
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=16,
        shuffle=False, num_workers=int(4),
        #collate_fn=BaiduCollate(opt.imgH, opt.imgW, keep_ratio=False)
    )
'''
    # root = './'
    # train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
    # test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)

    print(len(train_data))

    # get random 5 pixels per image and stack them all up as rgb values to get half a million random pixels
    pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(512*512, 3)[torch.randperm(512*512)[:5], :]
    #pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32*32, 3)[torch.randperm(32*32)[:5], :]
    px = torch.cat([pluck_rgb(x[2]) for x, y in train_data], dim=0).float()
    print(px.size())

    # load frames as list
    #frames = load_frames()
    #print('type:{}, shape:{}, range:[{}, {}]'.format(frames.dtype, frames.shape, np.min(frames), np.max(frames)))
    #frames[0].show()


    ncluster = 512
    with torch.no_grad():
        C = kmeans(px, ncluster, niter=8)

    print(C.size())


    # encode the training examples with our codebook to visualize how much we've lost in the discretization
    n_samples = 16
    ncol = 8
    nrow = n_samples // ncol + 1
    plt.figure(figsize=(20, 10))
    for i in range(n_samples):
            
        # encode and decode random data
        x, y = train_data[np.random.randint(0, len(train_data))]
        xpt = torch.from_numpy(np.array(x[2])).float().view(512*512, 3)
        #xpt = torch.from_numpy(np.array(x)).float().view(32*32, 3)        
        ix = ((xpt[:, None, :] - C[None, :, :])**2).sum(-1).argmin(1) # cluster assignments for each pixel
        
        # these images should look normal ideally
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(C[ix].view(512, 512, 3).numpy().astype(np.uint8))
        #plt.imshow(C[ix].view(32, 32, 3).numpy().astype(np.uint8))
        plt.axis('off')


    train_dataset = ImageDataset(train_data, C)
    #test_dataset = ImageDataset(test_data, C)
    train_dataset[0][0] # one example image flattened out into integers

if __name__ == '__main__':
    main()
