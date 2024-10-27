# 1) design model(input, output size, forward pass)
# 2) construct loss and optimizer
# 3) Training loop
#   - forward pass: compute predicition and loss
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets #for generating regression data set
import matplotlib.pyplot as plt

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs = 100
for epoch in range(n_epochs):
    # forward pass 
    y_predicted = model(X)
    # calculate loss
    loss = criterion(y_predicted, y)
    # backward pass
    loss.backward()
    # update weights
    optimizer.step()
    # clear gradient from last epoch
    model.zero_grad()

    if (epoch +1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
    
