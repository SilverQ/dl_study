import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

"""
[using tensorflow]
1. data define
2. weight define
3. hx, cost define
4. Optimizer define
5. session define
6. variables initialize
7. sess.run
"""


def linear_regression():
    x = [1., 2., 3.]
    y = [1., 2., 3.]

    W = tf.Variable(initial_value=[10.0], name='Weight')
    b = tf.Variable(initial_value=[10.0], name='bias')
    # print(W)
    # <tf.Variable 'Weight:0' shape=() dtype=float32_ref>
    # print(b)
    # <tf.Variable 'bias:0' shape=() dtype=float32_ref>

    # Our hypothesis XW+b
    # hypothesis = x * W + b
    hx = x * W + b
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # hx_val = sess.run(hx, feed_dict={xx: x})
    # print(hx_val)

    # cost/loss function
    cost = tf.reduce_mean(tf.square(hx - y))
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # cost_val = sess.run(cost, feed_dict={xx: x, yy: y})
    # print(cost_val)

    # optimizer
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    # Launch the graph in a session.
    with tf.Session() as sess:
        # Initializes global variables in the graph.
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            _, cost_val, w_val, b_val = sess.run([train, cost, W, b])
            if i % 1000 == 0:
                print(cost_val, w_val, b_val)
        hx_test = x * w_val + b_val
        print(hx_test)


# Using placeholder!!
x = [1., 2., 3.]
y = [1., 2., 3.]

W = tf.Variable(initial_value=[10.0], name='Weight')
b = tf.Variable(initial_value=[10.0], name='bias')

xx = tf.placeholder(dtype=tf.float32)

hx = xx * W + b
cost = tf.reduce_mean(tf.square(hx - y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        _, cost_val, w_val, b_val = sess.run([train, cost, W, b], feed_dict={xx: x})
        if i % 1000 == 0:
            print(cost_val, w_val, b_val)
    hx_test = sess.run(hx, feed_dict={xx: [4, 5, 6]})
    print(hx_test)
