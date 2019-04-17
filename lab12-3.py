import tensorflow as tf
import numpy as np


def long_rnn():
    sample = 'if you want to'
    # idx2char = set(sample)
    # print(idx2char)
    # {'t', 'w', ' ', 'n', 'o', 'u', 'y', 'a', 'i', 'f'}

    idx2char = list(set(sample))

    char2idx = {c: i for i, c in enumerate(idx2char)}
    print(char2idx)
    # {'i': 0, 't': 1, 'y': 2, 'o': 3, 'f': 8, ' ': 4, 'u': 5, 'n': 6, 'w': 7, 'a': 9}

    sample_idx = [char2idx[c] for c in sample]
    print(sample_idx)
    # [9, 4, 5, 7, 3, 2, 5, 6, 0, 8, 1, 5, 1, 3]

    seq_len = 5
    depth = len(idx2char)
    hidden_size = depth
    batch_size = 1
    print(depth)

    # x_data = sample_idx[:-1]    # 입력 : if you want t
    # y_data = sample_idx[1:]     # 출력 : f you want to
    x_data = [sample_idx[:-1]]    # []를 씌워줌으로써, x_data 추출시 rank를 2로 만들었다.
    y_data = [sample_idx[1:]]

    seq_len = len(x_data[0])
    print(seq_len)

    # print(x_data)
    # print(y_data)

    X = tf.placeholder(dtype=tf.int32, shape=[None, seq_len])
    Y = tf.placeholder(dtype=tf.int32, shape=[None, seq_len])

    X_one_hot = tf.one_hot(indices=X, depth=depth)

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    initial_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)


    # sequence_loss가 어떻게 동작하는지는 lab12-1에서 해보는 중
    outputs, _states = tf.nn.dynamic_rnn(rnn_cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

    weights = tf.ones([batch_size, seq_len])
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss=loss)

    pred = tf.argmax(outputs, axis=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(10):
            l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
            result = sess.run(pred, feed_dict={X: x_data})
            print(step, "loss: ", l, "prediction: ", result, "Y: ", y_data)

            result_str = [idx2char[c] for c in np.squeeze(result)]
            print("\tPrediction string: ", ''.join(result_str))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test = sess.run([X_one_hot, Y], feed_dict={X: x_data, Y: y_data})
        print(test)


long_rnn()

sentence = 'An open API (often referred to as a public API) is a publicly available application programming interface that provides developers with programmatic access to a proprietary software application or web service. APIs are sets of requirements that govern how one application can communicate and interact with another.'
# https://www.youtube.com/watch?v=2R6nfCNNz1U
# 9:25 진행 중
idx2char = list(set(sentence))
char2idx = {c: i for i, c in enumerate(idx2char)}
sample_idx = [char2idx[c] for c in sentence]

seq_len = 5
depth = len(idx2char)
hidden_size = depth
batch_size = 1

x_data = [sample_idx[:-1]]  # []를 씌워줌으로써, x_data 추출시 rank를 2로 만들었다.
y_data = [sample_idx[1:]]

seq_len = len(x_data[0])
print(seq_len)

X = tf.placeholder(dtype=tf.int32, shape=[None, seq_len])
Y = tf.placeholder(dtype=tf.int32, shape=[None, seq_len])
X_one_hot = tf.one_hot(indices=X, depth=depth)

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
initial_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

outputs, _states = tf.nn.dynamic_rnn(rnn_cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, seq_len])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss=loss)

pred = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(pred, feed_dict={X: x_data})
        print(step, "loss: ", l, "prediction: ", result, "Y: ", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction string: ", ''.join(result_str))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test = sess.run([X_one_hot, Y], feed_dict={X: x_data, Y: y_data})
    print(test)
