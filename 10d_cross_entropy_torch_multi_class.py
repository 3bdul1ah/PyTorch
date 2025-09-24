import torch
import torch.nn as nn
import numpy as np


loss = nn.CrossEntropyLoss()

# target is of size nBatch = 3
# each element has class label: 0, 1, or 2
y = torch.tensor([2, 0, 1])

# input is of size nBatch x nClasses = 3 x 3
# Y_pred are logits (not softmax)
y_pred_good = torch.tensor(
    [[0.1, 0.2, 3.9], # predict class 2
    [1.2, 0.1, 0.3], # predict class 0
    [0.3, 2.2, 0.2]]) # predict class 1

y_pred_bad = torch.tensor(
    [[0.9, 0.2, 0.1],
    [0.1, 0.3, 1.5],
    [1.2, 0.2, 0.5]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)

print(f"Good prediction with torch {type(y)}: {l1}\n \
Bad prediction with torch {type(y)}: {l2}")

# input tensor, dim int -- > tensor, index of tensor 
_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)
print(f"Max prediction with good: {predictions1}\n \
Bad prediction with bad: {predictions2}")
