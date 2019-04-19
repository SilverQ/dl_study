import tensorflow as tf
import numpy as np
np.set_printoptions(precision=1)

# Stacked RNN
input_sentence = 'An open API (often referred to as a public API) is a publicly available application programming interface that provides developers with programmatic access to a proprietary software application or web service. APIs are sets of requirements that govern how one application can communicate and interact with another.'

char_set = list(set(input_sentence))      # 입력된 문장을 char 단위로 구분하고 중복이 제거된 vocabulary를 생성.
char2idx = {c: i for i, c in enumerate(char_set)}   # char-index 쌍을 생성하여 dictionary에 저장.
encoded_sentence = [char2idx[c] for c in input_sentence]    # input sentence를 index로 encoding.
decoded_sentence = ''.join([char_set[t] for t in encoded_sentence]) # 원래 문장으로 decoding.
'''
print('char_set: ', char_set)
char_set:  ['d', 'I', ')', 'i', 'c', 'f', 'm', 'b', ' ', 'l', 'A', 'a', 'q', 'h', 'p', 's', 'u', 't', 'r', '(', 'P', 'v', 'g', 'n', 'w', 'e', '.', 'o', 'y']
print('char2idx: ', char2idx)
char2idx:  {'d': 0, 'h': 13, 'I': 1, 'p': 14, 'm': 6, ')': 2, 'i': 3, 's': 15, 'u': 16, 'o': 27, 'w': 24, 't': 17, 'r': 18, '(': 19, 'c': 4, 'f': 5, 'v': 21, 'e': 25, 'g': 22, 'b': 7, ' ': 8, 'n': 23, 'l': 9, 'A': 10, 'P': 20, 'q': 12, '.': 26, 'a': 11, 'y': 28}
print('sentence_encoded: ', sentence_encoded)
sentence_encoded:  [6, 16, 18, 12, 8, 10, 16, 18, 6, 22, 27, 18, 17, 12, 20, 3, 10, 16, 18, 4, 10, 20, 10, 4, 4, 10, 7, 18, 3, 12, 18, 25, 11, 18, 25, 18, 8, 9, 23, 24, 0, 14, 18, 6, 22, 27, 13, 18, 0, 11, 18, 25, 18, 8, 9, 23, 24, 0, 14, 24, 5, 18, 25, 28, 25, 0, 24, 25, 23, 24, 10, 18, 25, 8, 8, 24, 0, 14, 25, 3, 0, 12, 16, 18, 8, 4, 12, 1, 4, 25, 26, 26, 0, 16, 1, 18, 0, 16, 3, 10, 4, 20, 25, 14, 10, 18, 3, 19, 25, 3, 18, 8, 4, 12, 28, 0, 7, 10, 11, 18, 7, 10, 28, 10, 24, 12, 8, 10, 4, 11, 18, 15, 0, 3, 19, 18, 8, 4, 12, 1, 4, 25, 26, 26, 25, 3, 0, 14, 18, 25, 14, 14, 10, 11, 11, 18, 3, 12, 18, 25, 18, 8, 4, 12, 8, 4, 0, 10, 3, 25, 4, 5, 18, 11, 12, 20, 3, 15, 25, 4, 10, 18, 25, 8, 8, 24, 0, 14, 25, 3, 0, 12, 16, 18, 12, 4, 18, 15, 10, 23, 18, 11, 10, 4, 28, 0, 14, 10, 2, 18, 6, 22, 27, 11, 18, 25, 4, 10, 18, 11, 10, 3, 11, 18, 12, 20, 18, 4, 10, 21, 9, 0, 4, 10, 26, 10, 16, 3, 11, 18, 3, 19, 25, 3, 18, 1, 12, 28, 10, 4, 16, 18, 19, 12, 15, 18, 12, 16, 10, 18, 25, 8, 8, 24, 0, 14, 25, 3, 0, 12, 16, 18, 14, 25, 16, 18, 14, 12, 26, 26, 9, 16, 0, 14, 25, 3, 10, 18, 25, 16, 7, 18, 0, 16, 3, 10, 4, 25, 14, 3, 18, 15, 0, 3, 19, 18, 25, 16, 12, 3, 19, 10, 4, 2]
print('sentence_decoded: ', sentence_decoded)
sentence_decoded:  An open API (often referred to as a public API) is a publicly available application programming interface that provides developers with programmatic access to a proprietary software application or web service. APIs are sets of requirements that govern how one application can communicate and interact with another.
print(len(sentence_encoded), len(sentence_decoded))     # 314 314로 크기가 같은 것을 알 수 있다.
'''

# define parameters about encoding
input_len = len(input_sentence)     # 입력된 문장의 전체 shape 측정(이 lecture에서는 한문장(rank1)이라 길이만 측정)
embed_dim = len(char_set)     # word의 one hot vector embedding 사이즈, vocab size와 동일하면 OK.
num_classes = len(char_set)     # 분류기의 class 갯수. char-rnn에서는 다음 단어 예측이므로, embed_dim과 동일해야 함.
hidden_size = 128     # hidden size는 rnn cell의 hidden vector 크기.
'''
print('input_len: ', input_len)     # 314, 입력된 문장의 전체 길이. 
print('char_cnt: ', num_classes)    # 29
'''

# define parameters about training
seq_len = 10        # batch 하나 당 10글자씩 입력.
batch_size = 20     # 10글자씩 잘라진 corpus를 10개씩 묶어 1개의 batch input 생성.
epoch = 201         # 학습은 총 201회 수행.

# create batch
x_data = []
y_data = []

for i in range(input_len - seq_len):
    x_data.append(encoded_sentence[i:i + seq_len])  # list에 list가 append되는 형태로 rank 2 data가 생성됨.
    y_data.append(encoded_sentence[i + 1:i + seq_len + 1])
    # x_data.append(input_sentence[i:i + seq_len])  # encoding되기 전의 문자열을 잘라보면 이해가 빠르다.
    # y_data.append(input_sentence[i + 1:i + seq_len + 1])
'''
print(np.shape(x_data))
(305, 10), 314개의 character를 가진 문장을 10글자씩 잘라서 학습 데이터를 만드는 과정을 상상해보자.
크기가 10인 windows를 사용해 sentence를 한 글자씩 sliding하며 잘라낼 수 있는 조합의 수는
[문장의 총 길이] - [windows크기] + 1 = 305.
 예) 4글자를 2글자씩 잘라내보자. 1234 -> [1,2], [2,3], [3,4]
하지만 지금은 학습데이터를 만드는 중.
x에는 [1,2]를 입력 후 [2,3]을 예측하도록 할 것이며, [2,3] 입력 후 [3,4]를 예측시킬 것이다.
대응되는 y 라벨 데이터를 만들수 없기 때문에, [3,4]는 학습 데이터로 입력하지 못한다. 
가장 첫 쌍은 학습데이터로만 사용되고, 가장 마지막 쌍은 라벨로만 사용된다. 그 사이의 쌍은 학습에 한번 라벨에 한번씩 사용된다.
그 결과 range는 input_len - seq_len가 되어 304개씩의 학습 데이터와 라벨 데이터를 생성한다.
[[7, 4, 3, 23, 21, 17, 4, 3, 7, 1], [4, 3, 23, 21, 17, 4, 3, 7, 1, 0], ...
 ...
 [9, 5, 16, 3, 26, 4, 23, 5, 16, 17], [5, 16, 3, 26, 4, 23, 5, 16, 17, 2]]
print(x_data)     # (305, 10)
'''

# define parameters about iteration: window 크기에 맞게 잘라진 데이터를 한번 학습때 몇개씩 입력할지를 결정.
data_size = np.shape(x_data)[0]     # 304개의 학습 데이터가 만들어져 있음.
iter = np.int32(data_size / batch_size + 1)
# 20개씩을 한번에 학습시키면, 총 16번 iteration이 들어가야 함.
# 다만 마지막 iteration에서는 남아있는 15번 돌고 남은 4개만 입력될 것임.
print(np.shape(y_data))     # (304, 10)
print(data_size)    # 304
print(iter)         # 16

X = tf.placeholder(dtype=tf.int32, shape=[None, None])
Y = tf.placeholder(dtype=tf.int32, shape=[None, None])
X_one_hot = tf.one_hot(indices=X, depth=num_classes)
# print(X_one_hot)    # Tensor("one_hot:0", shape=(?, ?, 29), dtype=float32)
# tensor를 출력하면 위와 같이 shape, type을 보여준다. 앞으로는 shape만 보고 작업이 가능해야 한다.


def show_tensor(tensor, input):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor, feed_dict=input)
        print(result)


# show_tensor(X, {X: x_data[0:2], Y: y_data[1:3]})
# show_tensor(X_one_hot, {X: x_data[0:2], Y: y_data[1:3]})
# show_tensor(Y, {X: x_data[0:2], Y: y_data[1:3]})
'''
batch를 2개로 가정하여 x와 y를 출력해보자. one hot vector로 바꾸는 과정에서 rank가 증가한다.
[[22  1 25 21 28 17  1 25 22  9]
 [ 1 25 21 28 17  1 25 22  9 18]]
[[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
   0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 1.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]]
'''

rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)   # rnn cell을 define할 때는 hidden size만 입력하면 충분하다.
rnn_cell = tf.contrib.rnn.MultiRNNCell(cells=[rnn_cell]*2)  # rnn cell을 2층 구조로 쌓는다.
# initial_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
# 맨앞의 rnn cell에 들어가는 hidden state는 존재하지 않으므로, 위 명령을 사용해서 0을 입력

outputs1, _states = tf.nn.dynamic_rnn(rnn_cell, X_one_hot, dtype=tf.float32)
# outputs1, _states = tf.nn.dynamic_rnn(rnn_cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
# outputs2 = tf.reshape(outputs1, [batch_size, seq_len, num_classes])
# x_softmax = tf.reshape(outputs2, [-1, hidden_size])
# # print(outputs1)     # Tensor("rnn/transpose_1:0", shape=(20, ?, 29), dtype=float32)
# # print(outputs2)     # Tensor("Reshape:0", shape=(20, 10, 29), dtype=float32)
# # print(x_softmax)    # Tensor("Reshape_1:0", shape=(200, 29), dtype=float32)

# show_tensor(outputs1, {X: x_data[0:batch_size], Y: y_data[1:batch_size+1]})

# fc1 = tf.contrib.layers.fully_connected(x_softmax,
#                                         128,
#                                         activation_fn=tf.nn.relu)
# fc2 = tf.contrib.layers.fully_connected(fc1,
#                                         num_classes,
#                                         activation_fn=None)
# outputs = tf.reshape(fc2, [batch_size, seq_len, num_classes])
#
# weights = tf.ones([batch_size, seq_len])
# sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
# loss = tf.reduce_mean(sequence_loss)
# train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)
#
# pred = tf.argmax(outputs, axis=2)
#
# print('pred: ', pred)   # Tensor("ArgMax:0", shape=(20, 10), dtype=int64)
# show_tensor(pred, {X: x_data[0:batch_size], Y: y_data[1:batch_size+1]})
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for j in range(epoch):
#         for i in range(iter):
#             feed_dict = {X: x_data[i:i+batch_size], Y: y_data[i+1:i+batch_size+1]}
#             l, _ = sess.run([loss, train], feed_dict=feed_dict)
#             # print('epoch: ', j, 'iter: ', i, 'loss: ', l)
#         if j % 100 == 0:
#             prediction = sess.run(pred, feed_dict=feed_dict)
#             print('epoch: ', j)
#             print("prediction: ", prediction)
#     # prediction = sess.run(pred, feed_dict={X: x_data})
#     # for j, result in enumerate(prediction):
#     #     index = np.argmax(result, axis=1)
#     #     if j is 0:  # print all for the first result to make a sentence
#     #         print(''.join([char_set[t] for t in index]), end='')
#     #     else:
#     #         print(char_set[index[-1]], end='')

