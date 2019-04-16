import tensorflow as tf
import numpy as np

# 입력 텍스트를 신경써서 가공해보자.
# one hot encoding으로 바꿔주려면, vocabulary 생성, index가 필요

input_word = 'hihello'
id2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]
x_onehot = [[[1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0]]]
y_data = [[1, 0, 2, 3, 3, 4]]

batch_size = 1
seq_length = 6
input_dim = 5
hidden_size = 5
X = tf.placeholder(dtype=tf.float32, shape=[1, seq_length, input_dim])
Y = tf.placeholder(dtype=tf.int32, shape=[1, seq_length])

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
initial_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)


# sequence_loss가 어떻게 동작하는지는 lab12-1에서 해보는 중
outputs, _states = tf.nn.dynamic_rnn(rnn_cell, X, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, seq_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss=loss)

pred = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10):
        l, _ = sess.run([loss, train], feed_dict={X: x_onehot, Y: y_data})
        result = sess.run(pred, feed_dict={X: x_onehot})
        print(step, "loss: ", l, "prediction: ", result, "Y: ", y_data)

        result_str = [id2char[c] for c in np.squeeze(result)]
        print("\tPrediction string: ", ''.join(result_str))
