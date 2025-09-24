# 1. Import required libraries
import torch 
import torch.nn as nn

# 2. Prepare training data
# x are the input features (1D, shape [4, 1])
# y are the target labels (2x relation, shape [4, 1])
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# Input to test after training
x_test = torch.tensor([5], dtype=torch.float32)

# Get number of samples and features (for input size)
n_sampels, n_features = x.shape
print(n_sampels, n_features)

# 3. Design the model
#   - input_size = number of features
#   - output_size = same as input here (1)
input_size = n_features
output_size = n_features

# Create a linear regression model y = w*x + b
model = nn.Linear(input_size, output_size)

# Print model's prediction before training
print(f"Prediction before training: f(5) = {model(x_test).item():.3f}")


# 4. Set hyperparameters
learning_rate = 0.01
n_iters = 100

# Define loss function (Mean Squared Error)
loss = nn.MSELoss()

# Define optimizer (Stochastic Gradient Descent)
optmizier = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 5. Training Loop
for epoch in range(n_iters):
    # Forward pass: compute predicted y
    y_pred = model(x)
    
    # Compute loss between predicted y and actual y
    l = loss(y, y_pred)

    # Backward pass: compute gradients w.r.t weights & bias
    l.backward()

    # Update weights using optimizer
    optmizier.step()

    # Reset gradients to zero before next iteration
    optmizier.zero_grad()

    # Print progress every 10 epochs
    if epoch % 10 == 0:
        linear_layer = model.parameters()
        [w, b] = linear_layer 
        print(linear_layer)
        print(w.shape, "|" ,b.shape) # w is only 1 element so accesing it is [[0][0] as its size is 1,1 and b is 1 so [0]
        print("Weight:", w[0][0].item(), "Bias:", b[0].item())

        print(f"Epoch {epoch}: w = {w[0][0].item():.3f}, loss = {l:.6f}")

# Print model's prediction after training
print(f"Prediction after training: f(5) = {model(x_test).item():.3f}")
