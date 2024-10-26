import torch
import numpy as np

# f = w * x 
# f(actual) = 2 * x
# where w is the weight we are trying to find in training
# ultimately we want our model to adjust the weight to get closer and closer to 2 after each epoch

X = torch.tensor([1,2,3,4], dtype=torch.float32) # input data
Y = torch.tensor([2,4,6,8], dtype=torch.float32) # ground truths / actual output

w = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

# backpropogation(optimization algorithm)

# first pass: multiply each matrix element by our weight
def y_prediction(x):
    return x * w

# find Loss using square mean error (SME): (y' - y)**2.mean
def loss(y_predicted, y):
    return ((y_predicted - y)**2).mean()

# gradient: dLoss / dw:
#   1 / N * 2x * (ypred - y)
def gradient(y_predicted, y, x):
    return np.dot(2*x, y_predicted - y).mean()


# train 
learning_rate = 0.01
n_iter = 100

for epoch in range(n_iter):
    y_pred = y_prediction(X)

    l = loss(y_pred, Y)

    # dw = gradient(y_pred, Y, X)
    # same as:

    l.backward()

    # w -= learning_rate * dw
    # same as:
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'prediction: {y_pred} loss: {l:.3f} gradient: {w:.3f} weight: {w:.3f}')