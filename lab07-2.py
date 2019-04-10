import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random


np.set_printoptions(precision=2)
tf.set_random_seed(777)  # for reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# batch_xs, batch_ys = mnist.train.next_batch(100)    # 호출할 때마다 100개의 데이터를 읽어옴
# test_xs, test_ys = mnist.test()

# print(batch_xs)
# print(np.shape(batch_xs))       # (100, 784)
# print(np.shape(batch_ys))       # (100, 10)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784, 10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')

logits = tf.matmul(X, W) + b
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
cost = tf.reduce_sum(cost_i)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

pred = tf.argmax(logits, axis=1)
correct_pred = tf.equal(pred, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
'''
print(sess.run([pred[0], tf.argmax(Y, 1)[0]], feed_dict={X: batch_xs, Y: batch_ys}))
print(sess.run(cost_i, feed_dict={X: batch_xs, Y: batch_ys}))
print(sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys}))
'''

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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
            _, cost_val = sess.run(([train, cost]), feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / tot_batch
        acc_test = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        print(epoch, avg_cost, acc_test)
        # print(acc_test)

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
