# convolutional neural networks mainly work on image data
#   apply convolutional filters

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 4
batch_size = 4
learning_rate = 0.001
num_channels = 3 # 3 channels for each color now.
hidden_size = 6
filter_size = 5 # filter size (5x5)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download = True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self, num_channels, hidden_size, output_size):
        super(ConvNet, self).__init__()
        # output after convolution formula: ((W - F) + 2P) / (S + 1) 
        # w = input size, f = filter size, p = padding(zero in this case)
        # s = stride
        # => (32 - 5 + 0)/1 + 1 = 28 
        self.convL1 = nn.Conv2d(num_channels, hidden_size, filter_size)

        # will reduce size by two (pooling size)
        self.pooling = nn.MaxPool2d(2, 3)
        self.convL2 = nn.Conv2d(hidden_size, 16, filter_size)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc1 = nn.Linear(84, output_size)
        # self.relu = nn.ReLU()

    def forward(self, x):
        output = self.convL1(x)
        output = self.relu(output)
        output = self.pooling(output)
        output = self.convL2(output)

        return output

model = ConvNet(num_channels, hidden_size, 10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # convert images to write shape. images are in RGB so 32x32x3
        # essentially we're taking each image with 32 rows and 32 columns
        # and taking out the rows and columns to have one big sample list
        images = images.to(device)
        labels = labels.to(device)
        # images = images.reshape([-1, 32*32*3])
        # labels = images.reshape([-1, 32*32*3])
        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100:
            print(f'step {i + 1} / {n_total_steps} loss = {loss.item():.4f}')