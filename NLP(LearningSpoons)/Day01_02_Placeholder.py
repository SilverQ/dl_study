import tensorflow as tf
import torch
import keras
# from tensorflow.keras import preprocessing
from keras import preprocessing
# import os
import numpy as np

print(tf.__version__)  # 1.12.0
print(torch.__version__)  # 1.1.0
print(keras.__version__)  # 2.2.5

samples = ['너 오늘 이뻐 보인다',
           '나는 오늘 기분이 더러워',
           '끝내주는데, 좋은 일이 있나봐',
           '나 오늘 좋은 일이 생겼어',
           '아 진짜 짜증나',
           '오, 이거 진짜 좋은 것 같은데']
targets = [[1], [0], [1], [1], [0], [1]]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(samples)  # 토크나이저 학습은 전체 데이터를 대상으로!!
# 그러면 실 서비스 환경에서는 신조어들이 막 튀어나올텐데, 그때는 에러나나?

sequences = tokenizer.texts_to_sequences(samples)
sequences = preprocessing.sequence.pad_sequences(sequences, maxlen=6, padding='post')

targets = np.array(targets)

print("index text data : \n", sequences)
# index text data :
#  [[ 5  1  6  7  0  0]
#   [ 8  1  9 10  0  0]
#   [11  2  3 12  0  0]
#   [13  1  2  3 14  0]
#   [15  4 16  0  0  0]
#   [17 18  4  2 19 20]]
print("shape of sequences:", sequences.shape)  # shape of sequences: (6, 6)

word_index = tokenizer.word_index
print("index of each word : \n", word_index)
# index of each word :
#  {'오늘': 1, '좋은': 2, '일이': 3, '진짜': 4, '너': 5, '이뻐': 6, '보인다': 7, '나는': 8, '기분이': 9,
#   '더러워': 10, '끝내주는데': 11, '있나봐': 12, '나': 13, '생겼어': 14, '아': 15, '짜증나': 16, '오': 17,
#   '이거': 18, '것': 19, '같은데': 20}

print("targets: \n", targets)
# targets:
#  [[1]
#   [0]
#   [1]
#   [1]
#   [0]
#   [1]]
print("shape of targets:", targets.shape)  # shape of targets: (6, 1)

# # Original Code
# inputs_ph = tf.placeholder(dtype=tf.int32, shape=[6], name='sequences')
# labels_ph = tf.placeholder(dtype=tf.int32, shape=[1], name='labels')
#
# with tf.Session() as sess:
#     for i in range(len(sequences)):
#         sequence_input, label = sess.run([inputs_ph, labels_ph],
#                                          feed_dict={inputs_ph: sequences[i], labels_ph: targets[i]})
#         print('='*40)
#         print('Sequence input:', sequence_input)
#         print('Label:', label)
#     print('='*40)

# [Data Shuffling] : index를 랜덤한 리스트로 바꿔치기하여 셔플링을 시도해보자
# random_index = np.random.permutation(6)
# print(random_index)
# # [0 5 2 1 3 4], [0 2 5 1 4 3], ...
#
# inputs_ph = tf.placeholder(dtype=tf.int32, shape=[6], name='sequences')
# labels_ph = tf.placeholder(dtype=tf.int32, shape=[1], name='labels')
#
# with tf.Session() as sess:
#     for i in random_index:
#         sequence_input, label = sess.run([inputs_ph, labels_ph],
#                                          feed_dict={inputs_ph: sequences[i], labels_ph: targets[i]})
#         print('='*40)
#         print('Sequence input:', i, sequence_input)
#         print('Label:', label)
#     print('='*40)

# ========================================
# Sequence input: 0 [5 1 6 7 0 0]
# Label: [1]
# ========================================
# Sequence input: 2 [11  2  3 12  0  0]
# Label: [1]
# ========================================
# Sequence input: 4 [15  4 16  0  0  0]
# Label: [0]
# ========================================
# Sequence input: 3 [13  1  2  3 14  0]
# Label: [1]
# ========================================
# Sequence input: 1 [ 8  1  9 10  0  0]
# Label: [0]
# ========================================
# Sequence input: 5 [17 18  4  2 19 20]
# Label: [1]
# ========================================

# # [Make Batch] : 입력을 2개씩 잘라서 배치 학습을 시켜보자
# random_index = np.random.permutation(6)
# print(random_index)
# # [0 5 2 1 3 4], [0 2 5 1 4 3], ...
# shuffled_sequences = sequences[random_index]
# shuffled_targets = targets[random_index]
# print('shuffled_sequences: ', shuffled_sequences)
# batch_size = 2
#
# inputs_ph_b = tf.placeholder(dtype=tf.int32, shape=[None, 6], name='sequences')
# labels_ph_b = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels')
#
# with tf.Session() as sess:
#     for i in range(0, 6, 2):
#         feed_dict = {inputs_ph_b: shuffled_sequences[i:i+batch_size],
#                      labels_ph_b: shuffled_targets[i:i+batch_size]}
#         sequence_input, label = sess.run([inputs_ph_b, labels_ph_b],
#                                          feed_dict=feed_dict)
#         print('='*40)
#         print('Sequence input:', i, sequence_input)
#         print('Label:', label)
#     print('='*40)

# ========================================
# Sequence input: 0 [[ 5  1  6  7  0  0]
#  [15  4 16  0  0  0]]
# Label: [[1]
#  [0]]
# ========================================
# Sequence input: 2 [[13  1  2  3 14  0]
#  [ 8  1  9 10  0  0]]
# Label: [[1]
#  [0]]
# ========================================
# Sequence input: 4 [[11  2  3 12  0  0]
#  [17 18  4  2 19 20]]
# Label: [[1]
#  [1]]
# ========================================

# [Epoch] : 전체 입력을 2번 사용해서 반복 학습을 시도해보자
epochs = 2
# random_index = np.random.permutation(6)
# print(random_index)
# [0 5 2 1 3 4], [0 2 5 1 4 3], ...
# shuffled_sequences = sequences[random_index]
# shuffled_targets = targets[random_index]
# print('shuffled_sequences: ', shuffled_sequences)
batch_size = 2

inputs_ph_b = tf.placeholder(dtype=tf.int32, shape=[None, 6], name='sequences')
labels_ph_b = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels')

with tf.Session() as sess:
    for ep in range(epochs):
        random_index = np.random.permutation(6)
        shuffled_sequences = sequences[random_index]
        shuffled_targets = targets[random_index]
        for i in range(0, 6, 2):
            feed_dict = {inputs_ph_b: shuffled_sequences[i:i + batch_size],
                         labels_ph_b: shuffled_targets[i:i + batch_size]}
            sequence_input, label = sess.run([inputs_ph_b, labels_ph_b],
                                             feed_dict=feed_dict)
            print('=' * 40)
            print('Sequence input:', i, sequence_input)
            print('Label:', label)
        print('=' * 40)

# ========================================
# Sequence input: 0 [[13  1  2  3 14  0]
#  [17 18  4  2 19 20]]
# Label: [[1]
#  [1]]
# ========================================
# Sequence input: 2 [[ 8  1  9 10  0  0]
#  [15  4 16  0  0  0]]
# Label: [[0]
#  [0]]
# ========================================
# Sequence input: 4 [[11  2  3 12  0  0]
#  [ 5  1  6  7  0  0]]
# Label: [[1]
#  [1]]
# ========================================
# ========================================
# Sequence input: 0 [[ 8  1  9 10  0  0]
#  [17 18  4  2 19 20]]
# Label: [[0]
#  [1]]
# ========================================
# Sequence input: 2 [[11  2  3 12  0  0]
#  [13  1  2  3 14  0]]
# Label: [[1]
#  [1]]
# ========================================
# Sequence input: 4 [[ 5  1  6  7  0  0]
#  [15  4 16  0  0  0]]
# Label: [[1]
#  [0]]
# ========================================
