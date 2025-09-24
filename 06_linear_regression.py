# 1. Model Design (Input, Output Size, Forward Pass)
# 2. Constrcut loss and optimizer 
# 3. Training Loop
    # forward pass (aka predication and loss)
    # backward pass (aka gradients)
    # updaing the weights

import torch 
import torch.nn as nn 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0. data preparation
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1) 

n_sample, n_features = x.shape

# 1. model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2. loss an optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3. trainning loop
n = 100
for epoch in range(n):
    # forward pass and loss
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update weights 
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1}, loss = {loss.item():.4f}")

predicted = model(x).detach().numpy()

# plot 

plt.plot(x_numpy, y_numpy, "ro")
plt.plot(x_numpy, predicted, "b")
plt.show()
