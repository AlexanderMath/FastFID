import torch
from fid.fastfid import fastfid 

# Compute FID between CIFAR10 mini-batch and noise 
from cifar10 import images 
batch_size  = 64
noise       = torch.rand(images[:batch_size].shape)
fid         = fastfid(images[:batch_size], noise, gradient_checkpointing=True) 

# Construct adversarial examples for FID. 
noise   = noise.requires_grad_(True)
adam    = torch.optim.Adam([noise], 0.001)
steps   = images.shape[0]//batch_size 

for i in range(steps): 
    adam.zero_grad()
    fid = fastfid(noise, images[i*batch_size:(i+1)*batch_size], gradient_checkpointing=True) 
    fid.backward()
    adam.step()
    print("\r[%4i / %4i] fid=%4f"%(i+1, steps, fid), end="")
