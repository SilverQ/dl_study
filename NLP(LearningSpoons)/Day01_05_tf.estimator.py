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

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
sequences = preprocessing.sequence.pad_sequences(sequences, maxlen=6, padding='post')

targets = np.array(targets)
word_index = tokenizer.word_index

print("index text data : \n", sequences)
print("shape of sequences:", sequences.shape)
print('word index: ', word_index)
# word index:  {'오늘': 1, '좋은': 2, '일이': 3, '너': 4, '이뻐': 5, '보인다': 6, # '나는': 7, '기분이': 8, '더러워': 9,
#               '끝내주는데': 10, '있나봐': 11, '나': 12, '생겼어': 13, '아': 14, '진짜': 15, '짜증나': 16, '환상적인데': 17,
#               '정말': 18, '좋은거': 19, '같아': 20}

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
    # batch를 안하면 shape 에러가 뜨는데, 그냥 레코드 하나만 불러오나?, shape이 임베딩 크기라고 나온다.
    # ValueError: Input 0 of layer dense is incompatible with the layer:
    # : expected min_ndim=2, found ndim=1. Full shape received: [6]
    # dataset = dataset.shuffle(len(sequences))
    return dataset


vocab_size = len(word_index) + 1
emb_size = 128


def model_fn(features, labels, mode):
    print('features: ', features, '\n', 'labels: ', labels)
    embed_input = tf.keras.layers.Embedding(vocab_size, emb_size)(features)
    embed_input = tf.reduce_mean(embed_input, axis=-1)

    # with tf.Session() as sess:
    #     print('embed_input: ', sess.run(embed_input))
    # FailedPreconditionError (see above for traceback): GetNext() failed because the iterator has not been initialized.
    # Ensure that you have run the initializer operation for this iterator before getting the next element.
    # 	 [[node IteratorGetNext (defined at /home/hdh/Env/tensorflow-gpu-1.13/lib/python3.6/site-packages/tensorfl
    # 	 ow_estimator/python/estimator/util.py:110) ]]

    hidden_layer = tf.keras.layers.Dense(128, activation=tf.nn.relu)(embed_input)
    output_layer = tf.keras.layers.Dense(1)(hidden_layer)
    output = tf.nn.sigmoid(output_layer)
    # output = {'class': tf.cast(output, tf.int32)}

    # loss = tf.losses.mean_squared_error(output, labels)

    global_step = tf.train.get_global_step()  # 이거 하나가 새로 들어간건데,
    # train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step)

    # eval_metric_op = {"accuracy": tf.metrics.accuracy(labels, output)}

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

    # return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_op)
    # 출처: https://excelsior-cjh.tistory.com/157 [EXCELSIOR]


DATA_OUT_PATH = './data_out/'

if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)

estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=DATA_OUT_PATH + 'checkpoint/dnn')
estimator.train(input_fn=train_input_fn)
ev = estimator.evaluate(input_fn=eval_input_fn)
pred = estimator.predict(input_fn=pred_input_fn)

# train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
# eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
# tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# estimator.train_and_evaluate(input_fn=train_input_fn)

print('evaluation: ', ev)       # evaluation:  {'loss': 2.0537819e-07, 'global_step': 6570}
print('pred: ', pred)
print('pred: ', [int(round(item[0])) for item in list(pred)])
# print('evaluation: ', ev['loss'])       # evaluation:  {'loss': 2.0537819e-07, 'global_step': 6570}
# print('pred: ', pred)
# tf.logging.info(results)
