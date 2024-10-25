from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


t = tf.zeros([5,5,5,5]) #has 5^4 zeros. 
print(t)

t = tf.reshape(t, [125, -1]) #reshape size 
# print(t)

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# some learning algorithms:
    # linear regression
    # classification
    # clustering
    # hidden markov models

# linear regression: y = mx + b (2d). Find the line that best fits
#   a scatterplot of input data