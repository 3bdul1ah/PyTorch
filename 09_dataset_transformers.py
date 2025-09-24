import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math 

class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt(r'C:\Users\user\Desktop\PyTorch\data\wine.csv',
                        delimiter=',', dtype=np.float32, skiprows=1)

        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample =  self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor 

    def __call__(self, sample):
        inputs, targets = sample 
        inputs *= self.factor
        return inputs, targets
    

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

if __name__ == "__main__":
    
    print(" -------------Before Composed ----------------- ")
    dataset = WineDataset(transform=None)
    first_data = dataset[0]
    feature, labels = first_data
    print(type(feature), type(labels))
    print((feature), "| ", (labels))

    print(" -------------After Composed ----------------- ")
    dataset = WineDataset(transform=composed) # if transform = None ... it will be as it as numpy ,, or ToTensor .... or compopnse 
    first_data = dataset[0]
    feature, labels = first_data
    print(type(feature), type(labels))
    print((feature), "| ", (labels))



