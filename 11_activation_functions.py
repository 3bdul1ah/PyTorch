import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------
# Option 1: Using activation layers in __init__
# -------------------------
class NeuralNetwork_1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork_1, self).__init__() 
        self.linear1 = nn.Linear(input_size, hidden_size)   # first dense layer
        self.relu = nn.ReLU()                               # activation
        self.linear2 = nn.Linear(hidden_size, 1)            # second dense layer
        self.sigmoid = nn.Sigmoid()                         # output activation

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    

# -------------------------
# Option 2: Using functional API (torch / F)
# -------------------------
class NeuralNetwork_2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork_2, self).__init__() 
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))         
        out = torch.sigmoid(self.linear2(out))    
        return out
    

# -------------------------
# Main: train for 10 epochs + plot
# -------------------------
if __name__ == "__main__":
    # Hyperparameters
    input_size = 10      
    hidden_size = 5      
    batch_size = 4       
    learning_rate = 0.01
    num_epochs = 10      

    # Example dataset (random for demo)
    X = torch.randn(batch_size, input_size)               # features
    y = torch.randint(0, 2, (batch_size, 1)).float()      # binary labels

    # Choose a model
    model = NeuralNetwork_1(input_size, hidden_size)
    # model = NeuralNetwork_2(input_size, hidden_size)

    # Loss and optimizer
    criterion = nn.BCELoss()                              
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Track loss values
    loss_values = []

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss for plotting
        loss_values.append(loss.item())

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Final predictions
    print("\nFinal Predictions:", outputs.detach().numpy())

    # -------------------------
    # Plot training loss
    # -------------------------
    plt.figure(figsize=(6,4))
    plt.plot(range(1, num_epochs+1), loss_values, marker='o', linestyle='-', color='b')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
