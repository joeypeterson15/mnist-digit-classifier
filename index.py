import tensorflow as tf

t = tf.zeros([5,5,5,5]) #has 5^4 zeros. 
print(t)

t = tf.reshape(t, [125, -1]) #reshape size 
# print(t)

# some learning algorithms:
    # linear regression
    # classification
    # clustering
    # hidden markov models