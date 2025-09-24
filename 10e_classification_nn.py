import torch
import torch.nn as nn 



# Binary 
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.linear1 = nn.Linear(input_size,  hidden_size)
        self.reul = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1) # Only 1 output 


        def forward(self, x):
            linear1_layer_output = self.linear1(x)
            reul_layper_output = self.reul(linear1_layer_output)
            linar2_layer_output = self.linear2(reul_layper_output) 
            # need  softmax .... 

            y_pred = torch.sigmoid(linar2_layer_output)

            return y_pred 

model = NeuralNet1(input_size = 28*28, hidden_size = 5)
loss = nn.BCELoss()


# Multi Class 
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        self.linear1 = nn.Linear(input_size,  hidden_size)
        self.reul = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes) # multi class 

        def forward(self, x):
            linear1_layer_output = self.linear1(x)
            reul_layper_output = self.reul(linear1_layer_output)
            linar2_layer_output = self.linear2(reul_layper_output) 
            # no softmax .... 

            return linar2_layer_output 

model = NeuralNet2(input_size = 28*28, hidden_size = 5, num_classes = 3)
loss = nn.CrossEntropyLoss()