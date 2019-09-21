import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split

# base_path = './gdrive/My Drive/Colab Notebooks'
base_path = os.getcwd()
data_in_path = base_path + '/data_in/'
data_out_path = base_path + '/data_out/'
train_input_data = 'nsmc_train_input.npy'
train_label_data = 'nsmc_train_label.npy'
data_configs = 'data_configs.json'

input_data = np.load(open(data_in_path + train_input_data, 'rb'))
print('input_data: ', input_data[0:3])
label_data = np.load(open(data_in_path + train_label_data, 'rb'))
prepro_configs = json.load(open(data_in_path + data_configs, 'r'))

TEST_SPLIT = 0.1
RNG_SEED = 100   # 어제 실험한 것과 오늘 실험한게 일관성을 가지려면 초기값 고정 필요
VOCAB_SIZE = prepro_configs['vocab_size']
EMB_SIZE = 128
BATCH_SIZE = 128
NUM_EPOCHS = 2

input_train, input_eval, label_train, label_eval = train_test_split(input_data,
                                                                    label_data,
                                                                    test_size=TEST_SPLIT,
                                                                    random_state=RNG_SEED)


def mapping_fn(X, Y):
    inputs, label = {'x': X}, Y
    return inputs, label


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
    dataset = dataset.shuffle(buffer_size=len(input_train))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=NUM_EPOCHS)

    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_eval, label_eval))
    #     dataset = dataset.shuffle(buffer_size=len(input_eval))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)

    return dataset


def model_fn(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    # feature['x'] => (bs, 20)

    embedding_layer = tf.keras.layers.Embedding(
        VOCAB_SIZE,
        EMB_SIZE)(features['x'])  # (bs, 20, EMD_SIZE)

    dropout_emb = tf.keras.layers.Dropout(rate=0.5)(embedding_layer)  # (bs, 20, EMD_SIZE)

    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    for filter_size in filter_sizes:
        conv = tf.keras.layers.Conv1D(
            filters=100,
            kernel_size=filter_size,
            padding='valid',
            activation=tf.nn.relu,
            kernel_constraint=tf.keras.constraints.max_norm(3.))(dropout_emb)  # (bs, 20, 100)
        # 최대 norm 지정, weight clipping이 바로 이 부분

        pool = tf.keras.layers.GlobalMaxPool1D()(conv)  # [(bs, 100), (bs, 100), (bs, 100)]
        pooled_outputs.append(pool)

    h_pool = tf.concat(pooled_outputs, axis=1)  # (bs, 300)

    hidden = tf.keras.layers.Dense(units=250, activation=tf.nn.relu,
                                   kernel_constraint=tf.keras.constraints.max_norm(3.))(h_pool)  # (bs, 200)
    dropout_hidden = tf.keras.layers.Dropout(rate=0.5)(hidden, training=TRAIN)
    logits = tf.keras.layers.Dense(units=1)(dropout_hidden)  # sigmoid를 해주겠다  # (bs, 1)
    # logits = tf.keras.layers.Dense(units=2)(dropout_hidden)  # 이렇게하면 one-hot 필요

    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])  # (bs, 1)
        # labels = tf.one_hot(index = labels, depth=2)  # (bs, 2)

    if TRAIN:
        global_step = tf.train.get_global_step()
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        #         loss = tf.losses.softmax_cross_entropy(labels, logits)

        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)

    elif EVAL:
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        pred = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})

    elif PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'prob': tf.nn.sigmoid(logits),
            }
        )


tf.logging.set_verbosity(tf.logging.INFO)

est = tf.estimator.Estimator(model_fn, model_dir="data_out/checkpoint/yoon_kim")

est.train(train_input_fn)

valid = est.evaluate(eval_input_fn)

test_input_data = 'nsmc_test_input.npy'
test_label_data = 'nsmc_test_label.npy'

test_input_data = np.load(open(data_in_path + test_input_data, 'rb'))
test_label_data = np.load(open(data_in_path + test_label_data, 'rb'))


def test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((test_input_data, test_label_data))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


test_output = [pred['prob'] for pred in est.predict(test_input_fn)]

test_output = np.array(test_output)

print(test_output)
