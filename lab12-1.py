import tensorflow as tf
import numpy as np

hidden_size = 2

cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

x_data = np.array([[[1, 0, 0, 0]]], dtype=np.float32)

outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outputs_val = sess.run(outputs)
    print(outputs_val)
