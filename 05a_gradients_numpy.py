import numpy as np

# ----- Training Data -----
# x, y follow the simple relation y = 2x
x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

# ----- Model Parameter -----
# Weight initialized to zero
w = 0.0

# ----- Model -----
def forward(x):
    """Compute prediction ŷ = w * x"""
    return w * x

# ----- Loss Function -----
def loss(y, y_pred):
    """Mean Squared Error: L = (1/N) Σ (ŷ - y)²"""
    return ((y_pred - y) ** 2).mean()

# ----- Gradient Function -----
def gradient(x, y, y_pred):
    """Compute dL/dw = (2/N) Σ x * (ŷ - y)"""
    return np.dot(2 * x, (y_pred - y)) / len(x)

# ----- Before Training -----
print(f"Prediction before training: f(5) = {forward(5):.3f}")

# ----- Hyperparameters -----
learning_rate = 0.01
n_iters = 100  # use a larger number to see progress

# ----- Training Loop -----
for epoch in range(n_iters):
    # 1) Forward pass
    y_pred = forward(x)

    # 2) Compute loss
    l = loss(y, y_pred)

    # 3) Compute gradient
    dw = gradient(x, y, y_pred)

    # 4) Update weight (Gradient Descent)
    w -= learning_rate * dw

    # 5) Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: w = {w:.3f}, loss = {l:.6f}")

# ----- After Training -----
print(f"Prediction after training: f(5) = {forward(5):.3f}")
