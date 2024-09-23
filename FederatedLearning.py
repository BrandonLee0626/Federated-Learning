import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
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

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())

    global_count = sum([len(clients_trn_data[client_name]) for client_name in client_names])
    local_count = len(clients_trn_data[client_name])

    return local_count/global_count

def scale_model_parameter(parameter, scalar):
    return parameter*scalar

def sum_scaled_weights(scaled_weights_list):
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
    
global_model = Net().to(DEVICE)
criterion = nn.CrossEntropyLoss()

def train(model, train_loader):
    model.train()
    optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01, momentum=0.5)
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

def federated_learning(global_model, comm_rounds=100):
    for comm_round in range(comm_rounds):
        global_weight = {'fc1': global_model.fc1.weight.data, 'fc2':global_model.fc2.weight.data, 'fc3': global_model.fc3.weight.data}
        global_bias = {'fc1': global_model.fc1.bias.data, 'fc2': global_model.fc2.bias.data, 'fc3': global_model.fc3.bias.data}

        scaled_local_fc1_wieght_list = list()
        scaled_local_fc1_bias_list = list()
        scaled_local_fc2_wieght_list = list()
        scaled_local_fc2_bias_list = list()
        scaled_local_fc3_wieght_list = list()
        scaled_local_fc3_bias_list = list()

        client_names = list(clients_batched.keys())
        random.shuffle(client_names)

        for client_name in client_names:
            local_model = Net().to(DEVICE)
            local_model.fc1.weight.data = global_weight['fc1']
            local_model.fc1.bias.data = global_bias['fc1']
            local_model.fc2.weight.data = global_weight['fc2']
            local_model.fc2.bias.data = global_bias['fc2']
            local_model.fc3.weight.data = global_weight['fc3']
            local_model.fc3.bias.data = global_bias['fc3']

            train(local_model, clients_batched[client_name])

            scaling_factor = weight_scalling_factor(clients_batched, client_name)

            scaled_fc1_weight = scale_model_parameter(local_model.fc1.weight.data, scaling_factor)
            scaled_local_fc1_wieght_list.append(scaled_fc1_weight)
            scaled_fc2_weight = scale_model_parameter(local_model.fc2.weight.data, scaling_factor)
            scaled_local_fc2_wieght_list.append(scaled_fc2_weight)
            scaled_fc3_weight = scale_model_parameter(local_model.fc3.weight.data, scaling_factor)
            scaled_local_fc3_wieght_list.append(scaled_fc3_weight)
            
            scaled_fc1_bias = scale_model_parameter(local_model.fc1.bias.data, scaling_factor)
            scaled_local_fc1_bias_list.append(scaled_fc1_bias)
            scaled_fc2_bias = scale_model_parameter(local_model.fc2.bias.data, scaling_factor)
            scaled_local_fc2_bias_list.append(scaled_fc2_bias)
            scaled_fc3_bias = scale_model_parameter(local_model.fc3.bias.data, scaling_factor)
            scaled_local_fc3_bias_list.append(scaled_fc3_bias)
            
        average_fc1_weight = sum_scaled_weights(scaled_local_fc1_wieght_list)
        global_model.fc1.weight.data = average_fc1_weight
        average_fc1_bias = sum_scaled_weights(scaled_local_fc1_bias_list)
        global_model.fc1.bias.data = average_fc1_bias

        average_fc2_weight = sum_scaled_weights(scaled_local_fc2_wieght_list)
        global_model.fc2.weight.data = average_fc2_weight
        average_fc2_bias = sum_scaled_weights(scaled_local_fc2_bias_list)
        global_model.fc2.bias.data = average_fc2_bias

        average_fc3_weight = sum_scaled_weights(scaled_local_fc3_wieght_list)
        global_model.fc3.weight.data = average_fc3_weight
        average_fc3_bias = sum_scaled_weights(scaled_local_fc3_bias_list)
        global_model.fc3.bias.data = average_fc3_bias

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