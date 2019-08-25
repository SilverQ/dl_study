import tensorflow as tf
import torch
import keras
from keras import preprocessing
import os
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
word_index = tokenizer.word_index
# print("index text data : \n", sequences)
# print("shape of sequences:", sequences.shape)
# print("index of each word : \n", word_index)
# print("targets: \n", targets)
# print("shape of targets:", targets.shape)  # shape of targets: (6, 1)

# # [기본 코드]
# dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))    # 튜플로 감싸서 넣으면 tf.data가 알아서 잘라서 쓴다
# iterator = dataset.make_one_shot_iterator()           # 없어도 됨, If using `tf.estimator`, return the `Dataset` object directly from your input function
# next_data = iterator.get_next()
#
# with tf.Session() as sess:
#     while True:
#         try:
#             seq, lab = next_data
#             print(sess.run([seq, lab]))
#         except:
#             break        # 갯수가 끝나면 에러뜨면서 종료

# [셔플 데이터]
dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
dataset = dataset.shuffle(len(sequences))   # 데이터 길이만큼 지정해주는게 가장 좋아
iterator = dataset.make_one_shot_iterator()
next_data = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            seq, lab = next_data
            print(sess.run([seq, lab]))
        except:
            break

# # [배치 데이터]
# BATCH_SIZE = 2
#
# dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.shuffle(len(sequences))
# iterator = dataset.make_one_shot_iterator()
# next_data = iterator.get_next()
#
# with tf.Session() as sess:
#     while True:
#         try:
#             print(sess.run(next_data))
#         except tf.errors.OutOfRangeError:
#             break
#
# # Epochs
# BATCH_SIZE = 2
# EPOCH_SIZE = 2
#
# dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.shuffle(len(sequences))
# dataset = dataset.repeat(len(sequences))
#
# iterator = dataset.make_one_shot_iterator()
# next_data = iterator.get_next()
#
# with tf.Session() as sess:
#     while True:
#         try:
#             print(sess.run(next_data))
#         except:
#             break
#
# # [Map]
# def map_fn(X, Y=None):
#     inputs = {'x': X}
#     label = Y
#     return inputs, label
#
# dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
# dataset = dataset.map(map_fn)
# iterator = dataset.make_one_shot_iterator()
# next_data = iterator.get_next()
#
# with tf.Session() as sess:
#     while True:
#         try:
#             print(sess.run(next_data))
#         except:
#             break
#
# # [Map with two vairables]
# def map_fn(X1, X2, Y=None):
#     inputs = {'x1': X1, 'x2': X2}
#     label = Y
#     return inputs, label
#
#
# dataset = tf.data.Dataset.from_tensor_slices((sequences, sequences, targets))
# dataset = dataset.map(map_fn)
# iterator = dataset.make_one_shot_iterator()
# next_data = iterator.get_next()
#
# with tf.Session() as sess:
#     while True:
#         try:
#             print(sess.run(next_data))
#         except:
#             break
#
# # [Create All Features]
# BATCH_SIZE = 2
# EPOCH_SIZE = 2
#
# #############################################################
# # 지금까지 배운 API들을 사용해 본다.
# # 1. 주어진 데이터를 tf.data에 적용시킨다.
# # 2. map 사용한다.
# # 3. 배치 크기 만큼 가져온다.
# # 4. 순서를 섞어준다.
# # 5. 전체 데이터를 EPOCH SIZE 만큼 사용한다.
# # 6. 데이터를 하나씩 사용한다.
# # 7. iterator의 get_next를 통해 하나씩 가져오는 구조를 만든다.
# # 8. 세션을 구성해서 실행한다.
# #############################################################
#
# # ANSWER
# BATCH_SIZE = 2
# EPOCH_SIZE = 2
#
# def map_fn(X, Y=None):
#     inputs = {'x': X}
#     label = Y
#     return inputs, label
#
# dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
# dataset = dataset.map(map_fn)
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.shuffle(len(sequences))
# dataset = dataset.repeat(EPOCH_SIZE)
# iterator = dataset.make_one_shot_iterator()
# next_data = iterator.get_next()
#
# # ANSWER
# with tf.Session() as sess:
#     while True:
#         try:
#             print(sess.run(next_data))
#         except:
#             break

# epochs = 2
# batch_size = 2
#
# inputs_ph_b = tf.placeholder(dtype=tf.int32, shape=[None, 6], name='sequences')
# labels_ph_b = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels')
#
# with tf.Session() as sess:
#     for ep in range(epochs):
#         random_index = np.random.permutation(6)
#         shuffled_sequences = sequences[random_index]
#         shuffled_targets = targets[random_index]
#         for i in range(0, 6, 2):
#             feed_dict = {inputs_ph_b: shuffled_sequences[i:i + batch_size],
#                          labels_ph_b: shuffled_targets[i:i + batch_size]}
#             sequence_input, label = sess.run([inputs_ph_b, labels_ph_b],
#                                              feed_dict=feed_dict)
#             print('=' * 40)
#             print('Sequence input:', i, sequence_input)
#             print('Label:', label)
#         print('=' * 40)
