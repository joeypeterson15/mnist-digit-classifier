import torch
import numpy as np
import torch.nn as nn

# f = w * x 
# f(actual) = 2 * x
# where w is the weight we are trying to find in training
# ultimately we want our model to adjust the weight to get closer and closer to 2 after each epoch

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) # input data
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32) # ground truths / actual output

X_test = torch.tensor([5], dtype=torch.float32)

# w = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
#usually have to design what kind of model we'll use. 
# Linear is obvious for this example
n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
# model = nn.Linear(input_size, output_size)
model = LinearRegression(input_size, output_size)

learning_rate = 0.08
n_iter = 100

# backpropogation(optimization algorithm)

# first pass: multiply each matrix element by our weight. 
# manual model prediction
# def forward(x):
#     return x * w

# Manual Loss using mean square error (MSE): (y' - y)**2.mean
# def loss(y_predicted, y):
#     return ((y_predicted - y)**2).mean()

# pytorch MSE:
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# gradient: dLoss / dw:
#   1 / N * 2x * (ypred - y)
def gradient(y_predicted, y, x):
    return np.dot(2*x, y_predicted - y).mean()

# train 
for epoch in range(n_iter):
    y_pred = model(X)

    l = loss(Y, y_pred)

    # dw = gradient(y_pred, Y, X)
    # same as:

    l.backward()

    # w -= learning_rate * dw
    # same as:
    # with torch.no_grad(): #manual update weights
    #     w -= learning_rate * w.gradå

    # update weights with pytorch optimizer:
    optimizer.step()

    # w.grad.zero_() # when manual 
    optimizer.zero_grad() #with pytorch optimizer
    
    # zero gradients

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'loss: {l:.3f}, weight: {w[0][0].item():.3f}')

print(f'prediction after training: f(5) {model(X_test)}')