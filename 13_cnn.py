import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
batch_size = 4
learning_rate = 0.001


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # First conv layer: 3 RGB channels → 6 output channels, 5x5 kernel
        # Input: 32x32x3 → Output: 28x28x6 (32-5+1=28, no padding)
        # NOTE: Applies 6 different filters, each learns to detect different patterns
        # Each of the 6 filters is sized 5x5x3 (kernel_size × in_channels)
        # Result: 6 feature maps stacked together → 28x28x6 output
        # Docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        # First pooling: 2x2 max pooling with stride 2
        # Input: 28x28x6 → Output: 14x14x6 (size halved)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.relu1 = nn.ReLU()

        # Second conv layer: 6 input channels → 16 output channels, 5x5 kernel
        # Input: 14x14x6 → Output: 10x10x16 (14-5+1=10)
        self.conv2 = nn.Conv2d(6, 16, 5)
            
        # Second pooling: 2x2 max pooling with stride 2
        # Input: 10x10x16 → Output: 5x5x16 (size halved)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.relu2 = nn.ReLU()


        # Fully connected layers: flatten 5x5x16=400 features → 120 neurons
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400 → 120 (tunable)
        self.fc2 = nn.Linear(120, 84)           # 120 → 84 (tunable)
        self.fc3 = nn.Linear(84, 10)            # 84 → 10 classes (must match dataset)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu1(x)
        x = self.pool2(x)

        x = x.view(-1, 16 * 5 * 5)

        x = self.relu2(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)

        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
