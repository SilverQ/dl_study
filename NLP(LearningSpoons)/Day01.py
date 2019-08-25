import tensorflow as tf



"""
bert 돌리려면 16g gpu가 16개를 병렬로 돌려야 학습 가능. 그러다보니 정확도를 좀 떨어뜨리더라도 cnn을 채용해서 inference하는 경우가 많음
코사인, l1, l2, 자카드... 시밀러리티 측정 다양
분류모델의 유사도(true, flas), 유사도 계산
NLI가 분류모델을 사용한 유사도,

gluebenchmark.com
상대적으로 트랜스포머는 가벼워서 덤벼볼만

MT는 결과 비교가 쉽지 않다. 평가 방법은? BLUE score 사용
요즘은 뭘하던지 어테션
비트윈이라는 채팅앱은 연구용 사용동의르르 받아서 데잍터가 많다.
머행
ㅋㅋㅋ

우리는 토크나아징보다 모델, 예측 성능에 관심

텐서, 케라스, 토치?
주로 텐서케라스, 퓨어 케라스는 그닥
토치가 쉽기도 하지만 테서가 유용할 때도
텐서가 할 수 있는게 더 많음

토치는 중간중간 값을 출력해줌
텐서는 값을 뽑아볼 수는 있지만, 처음하는 사람에겐 어려울 수 있음

2.0에 와서는 eager execution을 지원하기 때문에 전혀 다른 텐서라고 봐도 되

estimator를 사용하면 eval, 추가학습 등이 간편
https://excelsior-cjh.tistory.com/157

하이레벨이라 간단, 모델 공유도 간단, 정확도 측정도 제각각인데, 모델 함수만 교환하고 estimator만 갖다붙이면 됨
그래프빌드, 학습, ㅇ벨, 에측, 배포, 체크포인트 등

pre-made estimator : 사전 정의된 뉴럴 네트워크 tf.estimator.DNNClassifier
사실 프리메이드는 써보기 수준이지, 상용화는 그닥...

custom estimator : 입력함수/모델함수 정의, instantiate estimator

입력함수 : feature를 딕셔너리 형태로
모델함수ㅜ는 상대적으로 규격화, 피처/라벨/모드를 필수입력, 파람스는 옵션
리턴은 tf.estimator.EstimatorSpec()dmfh


tf.data : estimator와 굉장히 잘 맞는 api
placeholder, feed_dict보다 높은 성능, 간단한 batch/epoch/shuffle/map 구햔
optimized pipeline


tensorflow module
모듈이 tf.nn, tf.layers, tf.keras 등에 각각 있어서 사용법도 다르고
근데 2.0에서는 tf.keras.layers 모듈로 통합예정
tf.layers는 tf.kerass.layer로 일대일 매칭


conv1d : 정방형 필터가 아님, 세로 길이는 입력과 같고, 가로 길이만 define, kernel_size=2 not (2,2)
-> i go to school은 4*dim일텐데, 2*2필터로 보는 것은 무의미, 따라서 2*dim 필터를 사용해 i g0, go to, to school 이렇게 스캔
https://missinglink.ai/guides/deep-learning-frameworks/keras-conv1d-working-1d-convolutional-neural-networks-keras/


reniew2@gmail.com


구글콜랩에서 tpu를 쓰랴먄 코드도 그에 맞게 작성해야 하는게 어려움

batch는 속도, robust를 위해

"""

import tensorflow as tf
import torch
import keras
from tensorflow.keras import preprocessing
from keras import preprocessing
import os
import numpy as np


print(tf.__version__)
# print(torch.__version__)
# print(torch.__version__)

samples = ['너 오늘 이뻐 보인다',
          '나는 오늘 기분이 더러워',
          '끝내주는데, 좋은 일이 있나봐',
          '나 오늘 좋은 일이 생겼어',
          '아 진짜 짜증나',
          '오, 이거 진짜 좋은 것 같은데']

targets = [[1], [0], [1], [1], [0], [1]]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
sequences = preprocessing.sequence.pad_sequences(sequences, maxlen=6, padding='post')

targets = np.array(targets)

print("index text data : \n", sequences)
print("shape of sequences:", sequences.shape)

word_index = tokenizer.word_index

print("index of each word : \n", word_index)

print("targets: \n", targets)
print("shape of targets:", targets.shape)

inputs_ph = tf.placeholder(dtype=tf.int32, shape=[6], name='sequences')
labels_ph = tf.placeholder(dtype=tf.int32, shape=[1], name='labels')

with tf.Session() as sess:
    for i in range(len(sequences)):
        sequence_input, label = sess.run([inputs_ph, labels_ph],
                                         feed_dict = {inputs_ph: sequences[i], labels_ph: targets[i]})
        print('='*40)
        print('Sequence input:', sequence_input)
        print('Label:', label)
    print('='*40)

# <Your code>

# with tf.Session() as sess:
#     for i in random_index:
#         # <Your code>
#         print('='*40)
#         print('Sequence input:', sequence_input)
#         print('Label:', label)
#     print('='*40)

