#
"""
20190904

app.monkeylearn.com
감성분성 메율레이트

SST-2는 SST-5에서 중립을 빼서 긍정 부정으로 구분한거

Specificity도 있네? Negative 중에서 정답 비율
불균형한 데이터에서는 중요한 척도겠군


윤킴. 학부논문이래 ㅎㅎ
Convolutional neural networks for sentence classification
성능이 은근히 괜찮아

n x k representation 단어를 임베딩
conv
max-over-time pooling
padding

dropout vs drop connection
혼동하는 경우가 많더라
웨이트를 계산 안하는게 드랍커넥션
https://www.tensorflow.org/api_docs/python/tf/nn/dropout
For each element of x, with probability rate, outputs 0, and otherwise scales up the input by 1 / (1-rate). The scaling is such that the expected sum is unchanged.
DROP된 비율에 따라 나머지 웨이트의 업데이트를 크게


https://www.aclweb.org/anthology/D14-1181
𝐫∈R^𝑚: ‘masking’ vector of Bernoulli random variable with probability p of being 1
‖𝑣‖_1=|𝑥_1+𝑥_2+…+𝑥_𝑛 | 이거 잘못된듯, 개별 elements에 절대값

Clipping weight by L2-norm
웨이트 자체가 얼마 이상 커지지 않도록 하겠다

필터수는 3,4,5크기짜리 100개
l2 weight clipping 3이하

CNN-rand
# random init(embedding matrix)
CNN-static
# pre-trained word2vec, non-training
CNN-non-static
# pre-trained word2vec, training
CNN-multichannel
# static nonstatic 둘다
# 나오는 과정이 모두 2개가 된다

Character-level Convolutional Networks for Text Classification
https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
정형화된 텍스트는 IFIDF가, 사용자리뷰같은건 캐릭터단위가 우수하더라


실습
EDA - 탐색적 데이터 분석
이건 뭘 하든 무조건 하는게 좋다
나의 고유의 데이터를 다룰때 특히

랜덤포레스트, xgboost

seaborn은 matplotlib위에서 돌아가서 더 이쁘다

https://github.com/e9t/nsmc/raw/master/ratings.txt
konlpy 만든분, 원래 java에서 돌던애라서,
mecab은 윈도우에서 안돌아가
twitter 형태소분석기는 okt로 이름이 바뀌었어

회사에서 띄어쓰기는 일단 버리고 시작하고
띄어쓰기 모듈을 이용해서 직접 해준다
사람들의 띄어쓰기는 충분히 믿을만하지 않다!!!



"""