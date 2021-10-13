import torch 
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset
dl      = DataLoader(torchvision.datasets.CIFAR10('/data/cifar', train=True, download=True))
images  = torch.from_numpy( dl.dataset.data / 255 ).cuda().float().permute([0,3,1,2])
