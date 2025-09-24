# 1. Design the model (inputs, output size, forward pass)
# 2. Construct loss and optimizer 
# 3. Training Loop
#   - Forward Pass: Computer Prediction
#   - Backward Pass: Gradients 
#   - Update Weights 


import torch 
import torch.nn as nn

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)


w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    """Compute prediction ŷ = w * x"""
    return w * x

print(f"Prediction before training: f(5) = {forward(5):.3f}")

learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optmizier = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):
    y_pred = forward(x)

    l = loss(y, y_pred)

    l.backward()

    optmizier.step()

    optmizier.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: w = {w:.3f}, loss = {l:.6e}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")
