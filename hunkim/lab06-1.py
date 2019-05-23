import tensorflow as tf
import numpy as np


# tensorflow를 사용해 softmax classifier를 구현해보자.

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([4, 3]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random_normal([3]), dtype=tf.float32, name='bias')

hx = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hx), axis=1))
# cost를 구할 때, y값과 hx 로그값의 곱의 합을 평균내야 함. -> axis = 1, 합하는 방향에 유의하자.
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hx), axis=0))

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run([-tf.reduce_sum(Y * tf.log(hx), axis=1), cost], feed_dict={X: x_data, Y: y_data}))

optimizer = tf.train.GradientDescentOptimizer(0.001)

train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(200000):
        _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
        if step % 10000 == 0:
            print(cost_val)
    pred = sess.run(tf.argmax(hx, axis=1), feed_dict={X: x_data, Y: y_data})
    print(pred)
