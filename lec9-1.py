import tensorflow as tf

# XOR 문제를 구현해보자.
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

W1 = [[5., -7.],
      [5., -7.]]
W2 = [[-11.],
      [-11.]]
b1 = [-8., 3.]
b2 = 6.

X = tf.to_float(X)

hx = tf.sigmoid(tf.matmul(X, W1) + b1)
XOR = tf.sigmoid(tf.matmul(hx, W2) + b2)

sess = tf.Session()
hx_val = sess.run(hx)
XOR_val = sess.run(XOR)

print(hx_val)
print(XOR_val)
