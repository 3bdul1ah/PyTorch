import torch
import torch.nn as nn
import numpy as np


loss = nn.CrossEntropyLoss()

# class 0
y  = torch.tensor([0]) # not one hot encoded anymore 
# nsamples x nclasses = 1 x 3 
# let's say:
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)

print(f"Good prediction with torch {type(y)}: {l1}\n \
Bad prediction with torch {type(y)}: {l2}")

# input tensor, dim int -- > tensor, index of tensor 
_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)
print(f"Max prediction with good: {predictions1}\n \
Bad prediction with bad: {predictions2}")
