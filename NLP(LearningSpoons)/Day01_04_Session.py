import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import preprocessing

samples = ['너 오늘 이뻐 보인다',
           '나는 오늘 기분이 더러워',
           '끝내주는데, 좋은 일이 있나봐',
           '나 좋은 일이 생겼어',
           '아 오늘 진짜 짜증나',
           '환상적인데, 정말 좋은거 같아']
targets = [[1], [0], [1], [1], [0], [1]]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
sequences = preprocessing.sequence.pad_sequences(sequences, maxlen=6, padding='post')

targets = np.array(targets)

word_index = tokenizer.word_index

print("index text data : \n", sequences)
print("shape of sequences:", sequences.shape)

inputs_ph = tf.placeholder(dtype=tf.int32, shape=[None, 6], name='sequences')
labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels')

epoch_size = 100
batch_size = 2

vocab_size = len(word_index) +1
emb_size = 128


# Embedding
embed_input = tf.keras.layers.Embedding(vocab_size, emb_size)(inputs_ph)
embed_input = tf.reduce_mean(embed_input, axis=-1)
# [bs, 6, dim] -> [bs, dim]으로 평균을 이용해 문장벡터로 변환

# Model
hidden_layer = tf.keras.layers.Dense(128, activation=tf.nn.relu)(embed_input)
output_layer = tf.keras.layers.Dense(1)(hidden_layer)
output = tf.nn.sigmoid(output_layer)

# # Loss
loss = tf.losses.mean_squared_error(output, labels_ph)

# Optimizer
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

shuffle_sequence = []
shuffle_label = []
index = 0

sess = tf.Session()
# print("start time : ", start_time)
sess.run(tf.global_variables_initializer())

for epoch in range(epoch_size):
    random_index = np.random.permutation(len(sequences))
    shuffled_sequences = sequences[random_index]
    shuffled_targets = targets[random_index]
    for i in random_index[::batch_size]:
        _, _loss = sess.run([train_op, loss], feed_dict={inputs_ph: shuffled_sequences[i:i+batch_size],
                                                         labels_ph: shuffled_targets[i:i+batch_size]})
        # _loss를 loss로 바꿔버리면, 그 다음 실행때 loss에 계산된 loss가 들어가버림!!
    if (epoch+1) % 10 == 0:
        print(f'# epoch{epoch} loss:{_loss}')

    shuffle_sequence.clear()
    shuffle_label.clear()
