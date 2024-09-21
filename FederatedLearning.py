import numpy as np
import random

import torch
import torch.nn
import torch.functional as F
from torchvision import transforms, datasets

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print("Using PyTorch version:", torch.__version__,' Device:', DEVICE)

BATCH_SIZE = int(input('BATCH_SIZE: '))
EPOCHS = int(input('EPOCHS: '))

train_dataset = datasets.MNIST(root = "../data/MNIST",
                               train = True,
                               download = True,
                               transform = transforms.ToTensor())

test_dataset = datasets.MNIST(root = "../data/MNIST",
                              train = False,
                              transform = transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False)

def creat_clients(num_clients=10, initial='clients'):
    client_names = [f'{initial}_{i+1}' for i in range(num_clients)]

    size = len(train_dataset) // num_clients
    shards = torch.utils.data.DataLoader(dataset = train_dataset,
                                         batch_size = size,
                                         shuffle = True)
    
    assert(len(shards) == len(client_names))

    return {client_names[i] : data for (i, data) in enumerate(shards)} 

clients = creat_clients()