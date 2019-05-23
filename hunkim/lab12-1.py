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


# lab12-2의 sequence_loss 해보는 중
y_data = tf.constant([[1, 1, 1]])
# pred = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)    # [[1, 0, 1]]
pred1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]], dtype=tf.float32)    # [[1, 0, 1]]
pred2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)    # [[1, 0, 1]]
weight = tf.constant([[1, 0, 0]], dtype=tf.float32)

# seq_loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weights)
seq_loss1 = tf.contrib.seq2seq.sequence_loss(logits=pred1, targets=y_data, weights=weight)
seq_loss2 = tf.contrib.seq2seq.sequence_loss(logits=pred2, targets=y_data, weights=weight)
with tf.Session() as sess:
    seq_loss_val = sess.run([seq_loss1, seq_loss2])
    print(seq_loss_val)     # 0.5967595, [0.5130153, 0.3711007]
