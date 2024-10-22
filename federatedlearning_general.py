# -*- coding: utf-8 -*-
"""FederatedLearning_general.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1y4FjQdKQncUELUTlo6vad4uvXJ2Zvkw-
"""

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

import matplotlib.pyplot as plt

DEVICE = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

print("Using PyTorch version:", torch.__version__,' Device:', DEVICE)

BATCH_SIZE = 32

train_dataset = datasets.MNIST(root = "../data/MNIST",
                               train = True,
                               download = True,
                               transform = transforms.ToTensor())

test_dataset = datasets.MNIST(root = "../data/MNIST",
                              train = False,
                              transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)

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

def parameter_scaling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())

    global_count = sum([len(clients_trn_data[client_name]) for client_name in client_names])
    local_count = len(clients_trn_data[client_name])

    return local_count/global_count

def scale_model_parameter(parameter, scalar):
    return parameter*scalar

def sum_scaled_parameters(scaled_weights_list):
    return sum(scaled_weights_list)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

criterion = nn.CrossEntropyLoss()

def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

def FedSGD(global_model, comm_rounds=10):
    for comm_round in range(comm_rounds):
      client_names = list(clients_batched.keys())
      random.shuffle(client_names)

      local_parameters = [list() for i in global_model.parameters()]

      for client_name in client_names:
        local_model = Net().to(DEVICE)
        local_optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.5)

        for key in global_model.state_dict().keys():
          local_model.state_dict()[key].copy_(global_model.state_dict()[key])

        train(local_model, clients_batched[client_name], local_optimizer)

        local_parameter = list(local_model.parameters())
        for i in range(len(local_parameter)):
          local_parameters[i].append(local_parameter[i])

      scaling_factor = parameter_scaling_factor(clients_batched, client_name)

      sum_parameters = list(map(sum, local_parameters))

      average_parameters = [local_parameter*scaling_factor for local_parameter in sum_parameters]
      
      for i, key in enumerate(global_model.state_dict().keys()):
        global_model.state_dict()[key].copy_(average_parameters[i].data)

def FedAvg(global_model, C, EPOCHS=3, comm_rounds=10):
    for comm_round in range(comm_rounds):
      m = int(max(C*len(clients_batched), 1))
      client_names = list(clients_batched.keys())
      client_names = random.sample(client_names, m)

      local_parameters = [list() for i in global_model.parameters()]

      for client_name in client_names:
        local_model = Net().to(DEVICE)
        local_optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.5)

        for key in global_model.state_dict().keys():
          local_model.state_dict()[key].copy_(global_model.state_dict()[key])
          
        for EPOCH in range(EPOCHS):
          train(local_model, clients_batched[client_name], local_optimizer)

        local_parameter = list(local_model.parameters())
        for i in range(len(local_parameter)):
          local_parameters[i].append(local_parameter[i])

      scaling_factor = parameter_scaling_factor(clients_batched, client_name)

      sum_parameters = list(map(sum, local_parameters))

      average_parameters = [local_parameter*scaling_factor for local_parameter in sum_parameters]

      for i, key in enumerate(global_model.state_dict().keys()):
        global_model.state_dict()[key].copy_(average_parameters[i].data)

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= (len(test_loader.dataset)/ BATCH_SIZE)
    test_accuracy = 100. * correct/len(test_loader.dataset)
    return test_loss, test_accuracy

baseline_test_accuracies = list()

for comm_rounds in range(0,100, 5):
  baseline_model = Net().to(DEVICE)

  FedSGD(baseline_model, comm_rounds=comm_rounds)
  test_loss, test_accuracy = evaluate(baseline_model, test_loader)
  baseline_test_accuracies.append(test_accuracy)

plt.plot(range(0, 100, 5), baseline_test_accuracies)
plt.xlabel('communication rounds')
plt.ylabel('test accuracy(%)')
plt.title('FedSGD_Generalized')

plt.show()

global_test_accuracies = list()

for comm_rounds in range(0,100, 5):
  global_model = Net().to(DEVICE)
  
  FedAvg(global_model, C=0.8, comm_rounds=comm_rounds)
  test_loss, test_accuracy = evaluate(global_model, test_loader)
  global_test_accuracies.append(test_accuracy)


plt.plot(range(0, 100, 5), global_test_accuracies)
plt.xlabel('communication rounds')
plt.ylabel('test accuracy(%)')
plt.title('FedAvg_Generalized')

plt.show()

