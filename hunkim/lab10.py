import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random


np.set_printoptions(precision=2)
tf.set_random_seed(777)  # for reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

initializer = tf.contrib.layers.xavier_initializer()
with tf.name_scope('layer1') as scope:
    # W1 = tf.Variable(tf.random_normal([784, 128]), name='weight1')
    W1 = tf.get_variable('weight1', shape=[784, 128],
                         # name=,
                         initializer=initializer)
    b1 = tf.get_variable('bias1', shape=[128], initializer=initializer)
    l1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    l1 = tf.nn.dropout(l1, keep_prob=keep_prob)
with tf.name_scope('layer2') as scope:
    # W2 = tf.Variable(tf.random_normal([128, 64]), name='weight2')
    # b2 = tf.Variable(tf.random_normal([64]), name='bias2')
    W2 = tf.get_variable('weight2', shape=[128, 64], initializer=initializer)
    b2 = tf.get_variable('bias2', shape=[64], initializer=initializer)
    l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)
    l2 = tf.nn.dropout(l2, keep_prob=keep_prob)

# W = tf.Variable(tf.random_normal([64, 10]), name='weight')
# b = tf.Variable(tf.random_normal([10]), name='bias')
W = tf.get_variable('weight', shape=[64, 10], initializer=initializer)
b = tf.get_variable('bisd', shape=[10], initializer=initializer)

logits = tf.matmul(l2, W) + b
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
cost = tf.reduce_sum(cost_i)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

pred = tf.argmax(logits, axis=1)
correct_pred = tf.equal(pred, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

training_epochs = 10
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        tot_batch = int(mnist.train.num_examples / batch_size)

        for i in range(tot_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, cost_val = sess.run(([train, cost]),
                                   feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
            avg_cost += cost_val / tot_batch
        acc_test = sess.run(accuracy,
                            feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})
        print(epoch, avg_cost, acc_test)
        # print(acc_test)

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(logits, 1),
                                   feed_dict={X: mnist.test.images[r:r+1], keep_prob: 1}))

    # plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    # plt.show()

# 1-layer result : 9 32.41425446423619 0.9108
# 3-layer, AdamOptimizer : 9 21.288263797760017 0.9413
# 3-layer, xavier initializer(weight1 only) : 9 11.298545000742783 0.9539
# 3-layer, xavier initializer(all weights and bias) : 9 6.7883265869929 0.9673
# add dropout : 9 15.371135563850405 0.9675
