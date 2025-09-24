import torch 
import torch.nn as nn
import numpy as np 

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
y = softmax(x)
print(f"softmax numpy {type(y)}: {y}")

x = torch.from_numpy(x)
y = torch.softmax(x, dim=0)
print(f"softmax tensor {type(y)}: {y}")



