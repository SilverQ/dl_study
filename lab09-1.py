import tensorflow as tf
import numpy as np


# XOR을 풀어보자

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [0, 1, 1, 0]

x_data = np.array(x_data, dtype=np.float32)
print(x_data)
y_data = np.reshape(y_data, newshape=[4, 1])
print(y_data)

X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W1 = tf.Variable(initial_value=tf.random_normal([2, 2]), dtype=tf.float32)
b1 = tf.Variable(tf.random_normal([2]), dtype=tf.float32)

W2 = tf.Variable(initial_value=tf.random_normal([2, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
hx = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost = tf.reduce_mean(tf.square(hx - Y))      # 나의 답
cost = - tf.reduce_mean(Y * tf.log(hx) + (1-Y) * tf.log(1 - hx))    # 김성훈 교수님 강의

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

pred = tf.cast(hx > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, Y), tf.float32))

# print(layer1)
# print(hx)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# layer1_val, hx_val = sess.run([layer1, hx], feed_dict={X: x_data})
# cost_val = sess.run(cost, feed_dict={X: x_data, Y: y_data})
# layer1_val = sess.run(layer1, feed_dict={X: x_data})
# layer1_val, hx_val = sess.run(layer1, hx)
# print(layer1_val)
# print(hx_val)
# print(cost_val)

for step in range(5001):
    _, cost_val, acc_val = sess.run([train, cost, acc], feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
        print(step, cost_val)

W1_val, b1_val, W2_val, b2_val = sess.run([W1, b1, W2, b2])
print(W1_val)
print(b1_val)
print(W2_val)
print(b2_val)
print(acc_val)

'''
나의 답
[[-3.3628094  2.9807374]
 [-5.5030484 -5.47557  ]]
[ 0.604157  -2.3299608]
[[-4.36692  ]
 [ 4.1570396]]
[-0.00185166]

cross entropy loss를 적용한 김성훈 교수님 코드
[[ 6.414344   5.603841 ]
 [-6.2512913 -5.8826017]]
[ 3.1729448 -2.9956114]
[[-9.331376]
 [ 9.94443 ]]
[4.30947]
 
'''