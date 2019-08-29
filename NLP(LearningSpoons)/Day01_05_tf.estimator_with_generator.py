import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import preprocessing

tf.logging.set_verbosity(tf.logging.INFO)

samples = ['너 오늘 이뻐 보인다',
           '나는 오늘 기분이 더러워',
           '끝내주는데, 좋은 일이 있나봐',
           '나 좋은 일이 생겼어',
           '오늘 좋은 일이 생겼어',
           '아 오늘 진짜 짜증나',
           '환상적인데, 정말 좋은거 같아']
targets = [[1], [0], [1], [1], [1], [0], [1]]


def read_sample(txt):
    for i in txt:
        yield i


# tmp = read_sample(samples)
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(read_sample(samples))
word_index = tokenizer.word_index
print('word_index:', word_index)

# word_index: {'오늘': 1, '좋은': 2, '일이': 3, '생겼어': 4, '너': 5, '이뻐': 6, '보인다': 7, '나는': 8, '기분이': 9, '더러워': 10,
#              '끝내주는데': 11, '있나봐': 12, '나': 13, '아': 14, '진짜': 15, '짜증나': 16, '환상적인데': 17, '정말': 18,
#              '좋은거': 19, '같아': 20}

sequences = tokenizer.texts_to_sequences(read_sample(samples))
sequences = preprocessing.sequence.pad_sequences(sequences, maxlen=6, padding='post')
print('sequences: ', sequences)

targets = np.array(targets)
# print(targets)

epoch_size = 10
batch_size = 2


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
    dataset = dataset.repeat(epoch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(len(sequences))
    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
    dataset = dataset.batch(batch_size)
    return dataset


def pred_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = dataset.batch(len(sequences))
    return dataset


vocab_size = len(word_index) + 1
emb_size = 128


def model_fn(features, labels, mode):
    print('features: ', features, '\n', 'labels: ', labels)
    embed_input = tf.keras.layers.Embedding(vocab_size, emb_size)(features)
    embed_input = tf.reduce_mean(embed_input, axis=-1)

    hidden_layer = tf.keras.layers.Dense(128, activation=tf.nn.relu)(embed_input)
    output_layer = tf.keras.layers.Dense(1)(hidden_layer)
    output = tf.nn.sigmoid(output_layer)
    global_step = tf.train.get_global_step()  # 이거 하나가 새로 들어간건데,

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(output, labels)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=output)
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.mean_squared_error(output, labels)
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step)
        eval_metric_op = {"accuracy": tf.metrics.accuracy(labels, output)}
        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss,
                                          eval_metric_ops=eval_metric_op,
                                          )
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=output
                                          )


DATA_OUT_PATH = './data_out/'

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)

estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=DATA_OUT_PATH + 'checkpoint/dnn')
estimator.train(input_fn=train_input_fn)
# ev = estimator.evaluate(input_fn=eval_input_fn)
# pred = estimator.predict(input_fn=pred_input_fn)
#
# print('evaluation: ', ev)       # evaluation:  {'loss': 2.0537819e-07, 'global_step': 6570}
# print('pred: ', pred)
# print('pred: ', [int(round(item[0])) for item in list(pred)])
