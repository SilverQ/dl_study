import tensorflow as tf
import numpy as np


def vanila_rnn():
    hidden_size = 2

    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

    x_data = np.array([[[1, 0, 0, 0]]], dtype=np.float32)

    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs_val = sess.run(outputs)
        print(outputs_val)

# lab12-2의 sequence_loss
y_data = tf.constant([[1, 1, 1]])
pred = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)
weight = tf.constant([[1, 1, 1]], dtype=tf.float32)

# seq_loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weights)
seq_loss = tf.contrib.seq2seq.sequence_loss(logits=pred, targets=y_data, weights=weight)
with tf.Session() as sess:
