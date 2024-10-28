# 1) design model(input, output_size, forward_pass)
# 2) construct loss and optimizer(also figure out which one to use)
# 3) Training loop
#   -forward pass: predict values and loss
#   -backward pass: gradients
#   -update weights

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt

# device config
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 1
batch_size = 100
learning_rate = 0.005

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           transform= transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          transform= transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle=False)
examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

# loss algorithm and activation functions
# good activation function is softmax
# A good loss algorithm is the cross function

class digitClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(digitClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.softmax = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # handle loss function in loop
        out = self.layer1(x)
        out = self.softmax(out) # activation function
        out = self.layer2(out)
        return out

model = digitClassifier(input_size, hidden_size, num_classes)

# need loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# print('model parameters:', model.parameters())
# print(f'train loader length: {len(train_loader)}')
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images shape: [100, 1, 28, 28]. 100 images with grayscale(1 channel) with 784(28x28) inputs

        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward
        pred = model(images)
        loss = criterion(pred, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max will give value, index. We want index(predicitions)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels.sum().item())
    
    print(f'correct n predictions: {n_correct}, n samples: {n_samples}')
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy={acc}')


