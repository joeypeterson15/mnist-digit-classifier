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
learning_rate = 0.005
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

print('data size:', train_loader)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self, output_size):
        super(ConvNet, self).__init__()
        # output after convolution formula: ((W - F) + 2P) / (S + 1) 
        # w = input size, f = filter size, p = padding(zero in this case)
        # s = stride
        self.convL1 = nn.Conv2d(3, 6, 5)
        self.pooling = nn.MaxPool2d(2, 2) # will reduce size by two (pooling size)
        self.convL2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.pooling(self.relu(self.convL1(x)))
        conv2 = self.pooling(self.relu(self.convL2(conv1)))
        conv2 = conv2.view(-1, 16*5*5)
        lin1 = self.relu(self.fc1(conv2))
        lin2 = self.relu(self.fc2(lin1))
        lin3 = self.fc3(lin2)
        return lin3

model = ConvNet(10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1000 == 0:
            print(f'step {i + 1} / {n_total_steps} loss = {loss.item():.4f}')