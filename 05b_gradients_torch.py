import torch 

# ----- Training Data -----
# x, y follow the simple relation y = 2x
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# ----- Model Parameter -----
# Weight initialized to zero, requires gradient for optimization
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# ----- Model -----
def forward(x):
    """Compute prediction ŷ = w * x"""
    return w * x

# ----- Loss Function -----
def loss(y, y_pred):
    """Mean Squared Error: L = (1/N) Σ (ŷ - y)²"""
    return ((y_pred - y) ** 2).mean()

# ----- Before Training -----
print(f"Prediction before training: f(5) = {forward(5):.3f}")

# ----- Hyperparameters -----
learning_rate = 0.01
n_iters = 100

# ----- Training Loop -----
for epoch in range(n_iters):
    # 1) Forward pass
    y_pred = forward(x)

    # 2) Compute loss
    l = loss(y, y_pred)

    # 3) Compute gradient: dl/dw
    l.backward()

    # 4) Update weight (Gradient Descent)
    with torch.no_grad():  # Disable autograd for weight update
        w -= learning_rate * w.grad

    # 5) Zero the gradient after each step
    w.grad.zero_()

    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: w = {w:.3f}, loss = {l:.6e}")

# ----- After Training -----
print(f"Prediction after training: f(5) = {forward(5):.3f}")
