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

X_img = tf.reshape(X, [-1, 28, 28, 1])      # 길이 784의 vector를 28x28 크기의 image 데이터로 변환.

initializer = tf.contrib.layers.xavier_initializer()
with tf.name_scope('conv1') as scope:
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.01))  # filter 변수, Variable임. 메모리 부족으로 32->16
    l1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # 옵션에 의해 결과 크기가 입력과 동일
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # [-1, 14, 14, 32] 출력

with tf.name_scope('conv2') as scope:
    W2 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01))     # 메모리 부족으로 64->32
    l2 = tf.nn.conv2d(l1, W2, strides=[1, 1, 1, 1], padding='SAME')  # 옵션에 의해 결과 크기가 입력과 동일
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # [-1, 7, 7, 64] 출력

l2 = tf.reshape(l2, [-1, 7*7*32])

# W = tf.Variable(tf.random_normal([64, 10]), name='weight')
# b = tf.Variable(tf.random_normal([10]), name='bias')
W = tf.get_variable('weight', shape=[7*7*32, 10], initializer=initializer)
b = tf.get_variable('bias', shape=[10], initializer=initializer)

logits = tf.matmul(l2, W) + b
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
cost = tf.reduce_sum(cost_i)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

pred = tf.argmax(logits, axis=1)
correct_pred = tf.equal(pred, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

training_epochs = 5
batch_size = 15     # 메모리 부족으로 100->15

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

# result : 4 1.9694227715590022 0.9857
