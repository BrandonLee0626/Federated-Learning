import numpy as np
import random

import torch
import torch.nn as nn
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

def batch_data(data_shard, BATCH_SIZE):
    dataset = torch.utils.data.TensorDataset(data_shard[0], data_shard[1])
    return torch.utils.data.DataLoader(dataset = dataset,
                                       batch_size = BATCH_SIZE,
                                       shuffle = True)

clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data, BATCH_SIZE=BATCH_SIZE)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def foward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
model = Net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()