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

epoch_size = 1000
batch_size = 2


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
    dataset = dataset.repeat(epoch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(len(sequences))
    return dataset


vocab_size = len(word_index) + 1
emb_size = 128


def model_fn(features, labels, mode):
    print(features)
    embed_input = tf.keras.layers.Embedding(vocab_size, emb_size)(features)
    embed_input = tf.reduce_mean(embed_input, axis=-1)

    hidden_layer = tf.keras.layers.Dense(128, activation=tf.nn.relu)(embed_input)
    output_layer = tf.keras.layers.Dense(1)(hidden_layer)
    output = tf.nn.sigmoid(output_layer)

    loss = tf.losses.mean_squared_error(output, labels)

    global_step = tf.train.get_global_step()  # 이거 하나가 새로 들어간건데,
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        # eval_metric=metric,
        # predictions=predictions
    )


DATA_OUT_PATH = './data_out/'

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)

estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=DATA_OUT_PATH + 'checkpoint/dnn')

tf.logging.set_verbosity(tf.logging.INFO)

estimator.train(train_input_fn)
