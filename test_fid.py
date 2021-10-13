"""

    Computes FID on cifar mini-batches using old and new algorithm.
    This should give the same values up to small numerical errors. 
    
    Expected output: 

        > python test_fid.py
        Loading inception... DONE! 2.6870s
        Files already downloaded and verified
        | Batch Size      | Prev Time       | New Time        | Prev FID        | New FID         |
        | 8               | 7.7308          | 0.5927          | 263.7568        | 263.7572        |
        | 16              | 7.5633          | 0.0615          | 267.7577        | 267.7598        |
        | 256             | 8.3497          | 0.7079          | 97.3997         | 97.3869         |

"""
import torchvision
import torch
from fid.fid import fid
from fid.fastfid import fastfid
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np 
import time 

if __name__ == "__main__":

    torch.set_grad_enabled(False)

    dl      = DataLoader(torchvision.datasets.CIFAR10('/data/cifar', train=True, download=True))
    tensor  = dl.dataset.data
    x_train = tensor 

    print("| %-15s | %-15s | %-15s | %-15s | %-15s |" % ("Batch Size", "Prev Time", "New Time", "Prev FID", "New FID") )

    for bs in [8, 16, 256]: 

        images1 = x_train[:bs].astype(np.float32)       / 255 
        images2 = x_train[bs:2*bs].astype(np.float32)   / 255 

        t0 = time.time()
        fid_new = fastfid(
            torch.from_numpy(images1).cuda().float().permute([0, 3, 1, 2]), 
            torch.from_numpy(images2).cuda().float().permute([0, 3, 1, 2]), 
            batch_size=bs)
        time_new = time.time() - t0 

        torch.cuda.synchronize() 

        t0 = time.time()
        fid_old = fid(images1, images2, batch_size=bs)
        torch.cuda.synchronize() 
        time_prev = time.time() - t0 

        print("| %-15.1i | %-15.4f | %-15.4f | %-15.4f | %-15.4f |" % (bs, time_prev, time_new, fid_old, fid_new) )
