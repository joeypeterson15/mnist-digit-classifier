import torch
import numpy as np

# f = w * x 
# f(actual) = 2 * x
# where w is the weight we are trying to find in training
# ultimately we want our model to adjust the weight to get closer and closer to 2 after each epoch

X = np.array([1,2,3,4], dtype=np.float32) # input data
Y = np.array([2,4,6,8], dtype=np.float32) # ground truths / actual output

w = 0.0

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
n_iter = 10

for epoch in range(n_iter):
    y_pred = y_prediction(X)

    curr_loss = loss(y_pred, Y)

    dw = gradient(y_pred, Y, X)

    w -= learning_rate * dw

    print(f'prediction: {y_pred} loss: {curr_loss:.3f} gradient: {dw:.3f} weight: {w:.3f}')