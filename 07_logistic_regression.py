import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 0 pre data
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target 

n_samples, n_features = x.shape
# print(n_samples ,"| ", n_features) # 569 |  30

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train = torch.from_numpy(x_train.astype(np.float32)).to(device)
x_test = torch.from_numpy(x_test.astype(np.float32)).to(device)
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1).to(device)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1).to(device)


# 1 setup model 
# f = wx + b, sigmoid function at the end 
class Logistic_Regression(nn.Module):
    def __init__(self, n_input_features):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = Logistic_Regression(n_features).to(device)

# 2 loss and optimizer
learning_rate = 0.01
loss = nn.BCELoss()
optimizer = SGD(model.parameters(), lr=learning_rate)

# 3 training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward 
    y_predicted = model(x_train)
    loss_value = loss(y_predicted, y_train)
    # backward
    loss_value.backward()

    # updates 
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss_value.item():.4f}")

# eval (if not used this it will be as computetianal graph and will be in traininng)
with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()
    accuracy = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) 
    print(f"Accuracy = {accuracy:.4f}")
