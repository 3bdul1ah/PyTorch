# MNSIT data
# DataLodaer, Transformation 
# Mltilayer Neural Net, activation function 
# Loss and optimizer 
# Training loop (batch training)
# Model evel 
# GPU Support 


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper param:
input_size = 784 # 28x28 image
hidden_size = 100 # can try another 
num_classes = 10
num_epochs = 2 
batch_size = 100
learning_rate = 0.01

train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

example = iter(train_loader)
samples, lables = next(example)
# print(f"samples {samples.shape} and lables {lables.shape}")

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()


class Neural_Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Neural_Network, self).__init__()
        self.L1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.L1(x)
        out = self.relu(out)
        out = self.L2(out)
        return out


# model,loss and optimizer 
model = Neural_Network(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, lables) in enumerate(train_loader):
        # 100, 1, 28, 28 
        # 100, 784

        images = images.reshape(-1, 28*28).to(device)
        lables = lables.to(device)

        # forward 
        outputs = model(images)
        loss = criterion(outputs, lables)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss {loss.item():.4f}')


# test 
with torch.no_grad():
    n_correct = 0
    n_samples = 0 
    for images, lables in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        lables = lables.to(device)

        outputs = model(images)
        # it return value and index (we need index)
        _, predications = torch.max(outputs, 1)
        n_samples += lables.shape[0]
        n_correct += (predications == lables).sum().item()
 
    accuracy = 100.0 * n_correct / n_samples
    print(f'accuracy = {accuracy}')
