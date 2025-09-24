import torch
import torch.nn as nn
import numpy as np


def cross_entropy(acual, predicted):
    loss =-np.sum(acual * np.log(predicted))
    return loss # / float(predicted.shape[0])

# class 0
y  = np.array([1, 0, 0])

# let's say:
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad =  np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)

print(f"Good prediction with numpy {type(y)}: {l1}\n \
Bad prediction with numpy {type(y)}: {l2}")


