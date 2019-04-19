import tensorflow as tf
import numpy as np
np.set_printoptions(precision=1)
import keras


# Stacked RNN

# input data
input_sentence = 'An open API (often referred to as a public API) is a publicly available application programming interface that provides developers with programmatic access to a proprietary software application or web service. APIs are sets of requirements that govern how one application can communicate and interact with another.'

# create index
char_set = list(set(input_sentence))
char2idx = {c: i for i, c in enumerate(sorted(char_set))}
idx2char = {char2idx[c]: c for c in char2idx}
'''
이 방식의 문제는 index가 생성할 때마다 바뀐다는 것. char-index가 매번 바뀐다면, 그에 따른 각 layer의 weight도 바뀌어야 할 것 같다.
따라서 index를 고정시킬 방안이 필요할 것으로 보인다. char_set을 정렬시켜서 사용해야 할 것 같다.
print(char_set)
print(char2idx)
print(idx2char)
['a', 'y', 'I', 'q', 'r', 'o', 'u', 'p', 'g', 'l', 'e', ' ', 'P', 'c', 'n', 'm', 'A', 'f', ')', 's', 'i', 'v', '.', 'h', 't', 'w', 'd', 'b', '(']
{'t': 24, 'a': 0, 'A': 16, ')': 18, 's': 19, 'i': 20, 'v': 21, 'q': 3, 'g': 8, ' ': 11, 'f': 17, 'P': 12, 'o': 5, '.': 22, 'u': 6, 'r': 4, 'p': 7, 'y': 1, 'h': 23, 'w': 25, 'l': 9, 'I': 2, 'e': 10, 'c': 13, 'n': 14, 'm': 15, 'd': 26, 'b': 27, '(': 28}
{0: 'a', 1: 'y', 2: 'I', 3: 'q', 4: 'r', 5: 'o', 6: 'u', 7: 'p', 8: 'g', 9: 'l', 10: 'e', 11: ' ', 12: 'P', 13: 'c', 14: 'n', 15: 'm', 16: 'A', 17: 'f', 18: ')', 19: 's', 20: 'i', 21: 'v', 22: '.', 23: 'h', 24: 't', 25: 'w', 26: 'd', 27: 'b', 28: '('}

['P', ' ', '(', 'o', 'u', 'a', 'l', 'g', '.', 'b', 'f', 'c', 'w', 'h', 'I', 'm', 's', 't', 'r', 'e', 'i', 'v', 'A', 'q', 'y', 'n', ')', 'p', 'd']
{'t': 17, 'y': 24, 'r': 18, 'P': 0, ' ': 1, 'e': 19, '(': 2, 'o': 3, 'u': 4, 'a': 5, 'l': 6, '.': 8, 'i': 20, 'b': 9, 'v': 21, 'A': 22, 'f': 10, 'q': 23, 'c': 11, 'n': 25, ')': 26, 'w': 12, 'h': 13, 'p': 27, 'I': 14, 'd': 28, 'g': 7, 'm': 15, 's': 16}
{0: 'P', 1: ' ', 2: '(', 3: 'o', 4: 'u', 5: 'a', 6: 'l', 7: 'g', 8: '.', 9: 'b', 10: 'f', 11: 'c', 12: 'w', 13: 'h', 14: 'I', 15: 'm', 16: 's', 17: 't', 18: 'r', 19: 'e', 20: 'i', 21: 'v', 22: 'A', 23: 'q', 24: 'y', 25: 'n', 26: ')', 27: 'p', 28: 'd'}

sorted(char_set) 적용 후
['d', 'I', 'i', 'A', 'o', 'l', 'r', ' ', 'w', 'p', 'm', 'n', '.', 'P', 'u', ')', 'h', 'b', 'v', 't', 'c', 'g', 's', '(', 'a', 'y', 'e', 'q', 'f']
{'d': 10, 'p': 20, 'u': 25, 'I': 5, 'b': 8, 'A': 4, ')': 2, 'o': 19, 'h': 14, 'l': 16, 'r': 22, ' ': 0, 'v': 26, 't': 24, 'c': 9, 'm': 17, 'n': 18, '.': 3, 'g': 13, 's': 23, '(': 1, 'P': 6, 'a': 7, 'y': 28, 'i': 15, 'e': 11, 'w': 27, 'q': 21, 'f': 12}
{0: ' ', 1: '(', 2: ')', 3: '.', 4: 'A', 5: 'I', 6: 'P', 7: 'a', 8: 'b', 9: 'c', 10: 'd', 11: 'e', 12: 'f', 13: 'g', 14: 'h', 15: 'i', 16: 'l', 17: 'm', 18: 'n', 19: 'o', 20: 'p', 21: 'q', 22: 'r', 23: 's', 24: 't', 25: 'u', 26: 'v', 27: 'w', 28: 'y'}

['g', '(', ')', 'I', 'f', 'y', 'u', 'e', 't', 'q', 'o', 'v', 'P', 'd', 's', 'r', 'm', ' ', 'l', 'i', 'b', 'w', 'n', '.', 'c', 'h', 'a', 'A', 'p']
{'d': 10, 'g': 13, '(': 1, 'r': 22, 'm': 17, ' ': 0, ')': 2, 'f': 12, 'y': 28, 'b': 8, 'e': 11, 'n': 18, 's': 23, 't': 24, 'i': 15, 'w': 27, 'I': 5, '.': 3, 'q': 21, 'c': 9, 'o': 19, 'h': 14, 'a': 7, 'v': 26, 'A': 4, 'p': 20, 'u': 25, 'l': 16, 'P': 6}
{0: ' ', 1: '(', 2: ')', 3: '.', 4: 'A', 5: 'I', 6: 'P', 7: 'a', 8: 'b', 9: 'c', 10: 'd', 11: 'e', 12: 'f', 13: 'g', 14: 'h', 15: 'i', 16: 'l', 17: 'm', 18: 'n', 19: 'o', 20: 'p', 21: 'q', 22: 'r', 23: 's', 24: 't', 25: 'u', 26: 'v', 27: 'w', 28: 'y'}

'''

# encoding sentence to index sequence
encoded_sentence = [char2idx[c] for c in input_sentence]
'''
print(encoded_sentence)
[10, 20, 15, 11, 22, 23, 20, 15, 10, 27, 25, 15, 21, 11, 1, 3, 23, 20, 15, 28, 23, 1, 23, 28, 28, 23, 9, 15, 3, 11, 15, 12, 19, 15, 12, 15, 22, 16, 8, 13, 5, 26, 15, 10, 27, 25, 0, 15, 5, 19, 15, 12, 15, 22, 16, 8, 13, 5, 26, 13, 2, 15, 12, 6, 12, 5, 13, 12, 8, 13, 23, 15, 12, 22, 22, 13, 5, 26, 12, 3, 5, 11, 20, 15, 22, 28, 11, 4, 28, 12, 14, 14, 5, 20, 4, 15, 5, 20, 3, 23, 28, 1, 12, 26, 23, 15, 3, 18, 12, 3, 15, 22, 28, 11, 6, 5, 9, 23, 19, 15, 9, 23, 6, 23, 13, 11, 22, 23, 28, 19, 15, 17, 5, 3, 18, 15, 22, 28, 11, 4, 28, 12, 14, 14, 12, 3, 5, 26, 15, 12, 26, 26, 23, 19, 19, 15, 3, 11, 15, 12, 15, 22, 28, 11, 22, 28, 5, 23, 3, 12, 28, 2, 15, 19, 11, 1, 3, 17, 12, 28, 23, 15, 12, 22, 22, 13, 5, 26, 12, 3, 5, 11, 20, 15, 11, 28, 15, 17, 23, 8, 15, 19, 23, 28, 6, 5, 26, 23, 24, 15, 10, 27, 25, 19, 15, 12, 28, 23, 15, 19, 23, 3, 19, 15, 11, 1, 15, 28, 23, 7, 16, 5, 28, 23, 14, 23, 20, 3, 19, 15, 3, 18, 12, 3, 15, 4, 11, 6, 23, 28, 20, 15, 18, 11, 17, 15, 11, 20, 23, 15, 12, 22, 22, 13, 5, 26, 12, 3, 5, 11, 20, 15, 26, 12, 20, 15, 26, 11, 14, 14, 16, 20, 5, 26, 12, 3, 23, 15, 12, 20, 9, 15, 5, 20, 3, 23, 28, 12, 26, 3, 15, 17, 5, 3, 18, 15, 12, 20, 11, 3, 18, 23, 28, 24]
sorted(char_set) 적용 후
[4, 18, 0, 19, 20, 11, 18, 0, 4, 6, 5, 0, 1, 19, 12, 24, 11, 18, 0, 22, 11, 12, 11, 22, 22, 11, 10, 0, 24, 19, 0, 7, 23, 0, 7, 0, 20, 25, 8, 16, 15, 9, 0, 4, 6, 5, 2, 0, 15, 23, 0, 7, 0, 20, 25, 8, 16, 15, 9, 16, 28, 0, 7, 26, 7, 15, 16, 7, 8, 16, 11, 0, 7, 20, 20, 16, 15, 9, 7, 24, 15, 19, 18, 0, 20, 22, 19, 13, 22, 7, 17, 17, 15, 18, 13, 0, 15, 18, 24, 11, 22, 12, 7, 9, 11, 0, 24, 14, 7, 24, 0, 20, 22, 19, 26, 15, 10, 11, 23, 0, 10, 11, 26, 11, 16, 19, 20, 11, 22, 23, 0, 27, 15, 24, 14, 0, 20, 22, 19, 13, 22, 7, 17, 17, 7, 24, 15, 9, 0, 7, 9, 9, 11, 23, 23, 0, 24, 19, 0, 7, 0, 20, 22, 19, 20, 22, 15, 11, 24, 7, 22, 28, 0, 23, 19, 12, 24, 27, 7, 22, 11, 0, 7, 20, 20, 16, 15, 9, 7, 24, 15, 19, 18, 0, 19, 22, 0, 27, 11, 8, 0, 23, 11, 22, 26, 15, 9, 11, 3, 0, 4, 6, 5, 23, 0, 7, 22, 11, 0, 23, 11, 24, 23, 0, 19, 12, 0, 22, 11, 21, 25, 15, 22, 11, 17, 11, 18, 24, 23, 0, 24, 14, 7, 24, 0, 13, 19, 26, 11, 22, 18, 0, 14, 19, 27, 0, 19, 18, 11, 0, 7, 20, 20, 16, 15, 9, 7, 24, 15, 19, 18, 0, 9, 7, 18, 0, 9, 19, 17, 17, 25, 18, 15, 9, 7, 24, 11, 0, 7, 18, 10, 0, 15, 18, 24, 11, 22, 7, 9, 24, 0, 27, 15, 24, 14, 0, 7, 18, 19, 24, 14, 11, 22, 3]
'''

# create batch
window_size = 5

x_data = []
y_data = []
for i in range(len(input_sentence)-window_size):
    x_data.append(encoded_sentence[i:i+window_size])
    y_data.append(encoded_sentence[i+1:i+window_size+1])
'''
    # x_data.append(input_sentence[i:i+window_size])
    # y_data.append(input_sentence[i+1:i+window_size+1])
print(x_data[-3:])
print(y_data[-3:])
.
.
.
print(x_data[:3])
print(y_data[:3])
['An op', 'n ope', ' open']
['n ope', ' open', 'open ']
.
.
.
['anoth', 'nothe', 'other']
['nothe', 'other', 'ther.']

[[4, 18, 0, 19, 20], [18, 0, 19, 20, 11], [0, 19, 20, 11, 18]]
[[18, 0, 19, 20, 11], [0, 19, 20, 11, 18], [19, 20, 11, 18, 0]]
.
.
.
[[7, 18, 19, 24, 14], [18, 19, 24, 14, 11], [19, 24, 14, 11, 22]]
[[18, 19, 24, 14, 11], [19, 24, 14, 11, 22], [24, 14, 11, 22, 3]]
'''
# print(np.shape(x_data))     # (309, 5)

X = tf.placeholder(dtype=tf.int32, shape=[None, window_size], name='input_data_x')
Y = tf.placeholder(dtype=tf.int32, shape=[None, window_size], name='label_data_y')

X_one_hot = tf.one_hot(indices=X, depth=len(char_set))
'''
print(X)
print(X_one_hot)
Tensor("input_data_x:0", shape=(?, 5), dtype=int32)
Tensor("one_hot:0", shape=(?, 5, 29), dtype=float32)
'''


def show_tensor(tensor, input):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tensor, feed_dict=input))

'''
show_tensor(X, {X: x_data[:5]})
[[ 4. 18.  0. 19. 20.]
 [18.  0. 19. 20. 11.]
 [ 0. 19. 20. 11. 18.]
 [19. 20. 11. 18.  0.]
 [20. 11. 18.  0.  4.]]

show_tensor(X_one_hot, {X: x_data[:2]})
[[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.
   0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]]]
'''

# create a BasicRNNCell
hidden_size = 5
basic_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([basic_rnn_cell] * 2, state_is_tuple=True)

# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

# defining initial state
batch_size = 3
# initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs, _states = tf.nn.dynamic_rnn(multi_rnn_cell, X_one_hot,
                                     # initial_state=initial_state,
                                     dtype=tf.float32)

# print(x_data[:2])
# show_tensor(outputs, {X: x_data[:2]})

# weights = tf.ones([batch_size, hidden_size])
# sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
# loss = tf.reduce_mean(sequence_loss)
# train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)
#
# pred = tf.argmax(outputs, axis=2)

show_tensor(outputs, {X: x_data[:3], Y: y_data[:3]})
# show_tensor(pred, {X: x_data[:2], Y: y_data[:2]})
# show_tensor(initial_state, {X: x_data[:2]})
