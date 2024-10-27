# 1) design model(input, output_size, forward_pass)
# 2) construct loss and optimizer(also figure out which one to use)
# 3) Training loop
#   -forward pass: predict values and loss
#   -backward pass: gradients
#   -update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler #scale our features
from sklearn.model_selection import train_test_split
# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(f'samples/features: {n_samples} {n_features}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#scale
sc = StandardScaler()

# 1) setup model
# 2) setup loss and optimizer
# 3) training loop