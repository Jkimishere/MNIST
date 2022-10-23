import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
from torchvision import transforms, datasets



batch_size = 10

train_data = datasets.MNIST('./data', True, transform=transforms.ToTensor(),download=True)
train_loader = utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


test_data = datasets.MNIST('./data', False, transform=transforms.ToTensor(),download=True)
test_loader = utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)



