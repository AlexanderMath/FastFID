# Fast (MiniBatch) Fréchet Inception Distance (FID)
Compute (and backpropagate) fast through FID. 
You can start by using the <a href="https://github.com/AlexanderMath/FastFID/blob/main/minimal_working_example.py">minimal working example</a> below or <a href="https://colab.research.google.com/drive/1PTI9Nwl0BkJsEt7dsOJUklvrk56YhnWc?usp=sharing">this notebook</a> which fine-tunes BigGAN to minimize FID. 

## Minimal Working Example 
The code below uses a function ```fid(imagesA, imagesB)``` supporting backpropagating wrt ```imagesA```.

```
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
```

Note. You can make it faster by precomputing the mean and standard deviation of the Inception activations for imagesB. I chose to share the above slower code because it works for any dataset out of the box (without precomputing anything). 

## Test Code

```
> python test_fid.py 
| Batch Size      | Prev Time       | New Time        | Prev FID        | New FID         |
| 8               | 13.9932         | 0.0560          | 263.7587        | 263.7572        |
| 16              | 14.1004         | 0.0632          | 267.7613        | 267.7590        |
| 32              | 13.9612         | 0.1125          | 217.7122        | 217.7117        |

> python test_sqrtm.py

```
 > python test_sqrt.py 
        | Batch Size  | SciPy Time  | New Time    | Real Tr     | SciPy Tr    | New Tr      | SciPy Error | New Error   |
        | 8           | 4.2649      | 0.3275      | 14337.4844  | 14513.4677  | 14337.4854  | -175.9833   | -0.0010     |
        | 16          | 4.4771      | 0.0022      | 30551.6914  | 30777.3852  | 30551.6895  | -225.6938   | 0.0020      |
        | 32          | 4.4943      | 0.0029      | 64191.7812  | 64496.6673  | 64191.7812  | -304.8860   | 0.0000      |
        | 64          | 5.0165      | 0.0040      | 129579.5312 | 129989.2554 | 129579.5156 | -409.7242   | 0.0156      |
        | 128         | 4.3869      | 0.0072      | 259934.5469 | 260500.5697 | 259934.5625 | -566.0228   | -0.0156     |
```


## Cite us 
```
@article{mathiasen2020backpropagating,
  title={Backpropagating through Fr$\backslash$'echet Inception Distance},
  author={Mathiasen, Alexander and Hvilsh{\o}j, Frederik},
  journal={arXiv preprint arXiv:2009.14075},
  year={2020}
}
```



