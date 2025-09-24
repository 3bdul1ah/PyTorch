import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math 

class WineDataset(Dataset):
    def __init__(self):
        # dataset loading
        xy = np.loadtxt(r'C:\Users\user\Desktop\PyTorch\data\wine.csv',
                        delimiter=',', dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


if __name__ == "__main__":

    dataset = WineDataset()
    # first_data = dataset[0]
    # features, label = first_data
    # print(features, label)

    dataloader = DataLoader(dataset=dataset, batch_size=4, 
                            shuffle=True, num_workers=2)

    dataiter = iter(dataloader)   # create an iterator
    features, labels = next(dataiter)   # get the first batch
    print(features, labels)

    # trainnig loop:
    num_epochs = 2 
    total_samples = len(dataset)
    n_iter = math.ceil(total_samples / 4)

    print(total_samples, "| ", n_iter)

    for epoch in range(num_epochs):
        for i, (input, labels) in enumerate(dataloader):
            # forward

            # backward 

            # update 

            # DUMMY:
            if (i+1) % 5 == 0:
                print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iter}, inputs {input.shape}")

# datasets from pytorch ... :

#  -torchvision.datasets.MNIST()
# fashion-mnist, cifar, coco

