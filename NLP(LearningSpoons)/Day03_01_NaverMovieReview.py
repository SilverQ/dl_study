import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
import konlpy
# from konlpy import twitter
import re
import json
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
# %matplotlib inline

cur_path = os.getcwd()
# print('cur_path: ', cur_path)
data_in_path = cur_path + '/data_in/'
data_out_path = cur_path + '/data_out/'

print("파일 크기 : ")
for file in os.listdir(data_in_path):
    if 'txt' in file :
        print('  *', file.ljust(30) + str(round(os.path.getsize(data_in_path + file) / 1000000, 2)) + 'MB')

train_data = pd.read_csv(data_in_path + 'ratings_train.txt', header=0, delimiter='\t')
print(train_data.head(3))

train_length = train_data['document'].astype(str).apply(len)
print(train_length.head(3))
# text의 길이
print(f'전체 학습데이터의 개수: {len(train_data)}')

train_word_counts = train_data['document'].astype(str).apply(lambda x: len(x.split(' ')))


def plot_data():
    # 그래프에 대한 이미지 사이즈 선언
    plt.figure(figsize=(12, 5))  # canvas 생성, figsize: (가로, 세로) 형태의 튜플로 입력
    # 히스토그램 선언
    #   - bins: 히스토그램 값들에 대한 버켓 범위,     range: x축 값의 범위,    alpha: 그래프 색상 투명도
    #   - color: 그래프 색상,                     label: 그래프에 대한 라벨
    plt.hist(train_length, bins=140, alpha=0.5, color='r', label='word')
    # plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of length of review')      # 그래프 제목
    plt.xlabel('Length of review')                      # 그래프 x 축 라벨
    plt.ylabel('Number of review')                      # 그래프 y 축 라벨
    plt.savefig('Histogram of length of review.png')

    # 그래프에 대한 이미지 사이즈 선언
    # figsize: (가로, 세로) 형태의 튜플로 입력
    plt.figure(figsize=(12, 5))  # canvas 생성
    # 히스토그램 선언
    # bins: 히스토그램 값들에 대한 버켓 범위
    # range: x축 값의 범위
    # alpha: 그래프 색상 투명도
    # color: 그래프 색상
    # label: 그래프에 대한 라벨
    plt.hist(train_length, bins=140, alpha=0.5, color= 'r', label='word')
    plt.yscale('log', nonposy='clip')
    # 그래프 제목
    plt.title('Log-Histogram of length of review')
    # 그래프 x 축 라벨
    plt.xlabel('Length of review')
    # 그래프 y 축 라벨
    plt.ylabel('Number of review')
    plt.savefig('Log-Histogram of length of review.png')

    # print(f'리뷰 길이 최대 값: {np.max(train_lenght)}'.format())   # 이거 틀린거
    print(f'리뷰 길이 최대 값: {np.max(train_length)}')

    print(f'리뷰 길이 최소 값: {np.min(train_length)}')
    print(f'리뷰 길이 평균 값: {np.mean(train_length):.2f}')
    print(f'리뷰 길이 표준편차: {np.std(train_length):.2f}')
    print(f'리뷰 길이 중간 값: {np.median(train_length)}')
    # 사분위의 대한 경우는 0~100 스케일로 되어있음
    print(f'리뷰 길이 제 1 사분위: {np.percentile(train_length, 25)}')
    print(f'리뷰 길이 제 3 사분위: {np.percentile(train_length, 75)}')

    plt.figure(figsize=(12, 5))
    # 박스플롯 생성
    # 첫번째 파라메터: 여러 분포에 대한 데이터 리스트를 입력
    # labels: 입력한 데이터에 대한 라벨
    # showmeans: 평균값을 마크함

    plt.boxplot(train_length,
                 labels=['counts'],
                 showmeans=True)
    # None   # 깔끔하게 그림만 출력하고 싶으면 쓰자
    plt.savefig('Box-plot of length of review.png')

    train_review = [review for review in train_data['document'] if type(review) is str]

    font_path = './NanumBarunGothic.ttf'

    wordcloud = WordCloud(font_path=font_path).generate(' '.join(train_review))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('WordCloud of review.png')

    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(train_data['label'])
    plt.savefig('CountPlot of train data.png')

    print(f"긍정 리뷰 개수: {train_data['label'].value_counts()[1]}")
    print(f"부정 리뷰 개수: {train_data['label'].value_counts()[0]}")

    # train_word_counts = train_data['document'].astype(str).apply(lambda x:len(x.split(' ')))

    print(train_data.head())
    print(train_word_counts[:3])

    plt.figure(figsize=(15, 10))
    plt.hist(train_word_counts, bins=40, facecolor='r',label='train')
    plt.title('Log-Histogram of word count in review', fontsize=15)
    # plt.yscale('log', nonposy='clip')
    plt.legend()
    plt.xlabel('Number of words', fontsize=15)
    plt.ylabel('Number of reviews', fontsize=15)
    plt.savefig('Number of words.png')


# plot_data()

# print(f'리뷰 단어 개수 최대 값: {np.max(train_word_counts)}'.format())
# print(f'리뷰 단어 개수 최소 값: {np.min(train_word_counts)}')
# print(f'리뷰 단어 개수 평균 값: {np.mean(train_word_counts):.2f}')
# print(f'리뷰 단어 개수 표준편차: {np.std(train_word_counts):.2f}')
# print(f'리뷰 단어 개수 중간 값: {np.median(train_word_counts)}')
# # 사분위의 대한 경우는 0~100 스케일로 되어있음
# print(f'리뷰 단어 개수 제 1 사분위: {np.percentile(train_word_counts, 25)}')
# print(f'리뷰 단어 개수 제 3 사분위: {np.percentile(train_word_counts, 75)}')

qmarks = np.mean(train_data['document'].astype(str).apply(lambda x: '?' in x))  # 물음표가 구두점으로 쓰임
fullstop = np.mean(train_data['document'].astype(str).apply(lambda x: '.' in x))  # 마침표

print('물음표가있는 질문: {:.2f}%'.format(qmarks * 100))
print('마침표가 있는 질문: {:.2f}%'.format(fullstop * 100))

review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", train_data['document'][4])
print(review_text)

okt = Okt()
review_text = okt.morphs(review_text, stem=True)  # stemming
print(review_text)

stop_words = set(['은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한'])
clean_review = [token for token in review_text if not token in stop_words]
print(clean_review)
# nltk같은데에는 불용어를 제공할 것이야, 휴리스틱한 부분


def preprocessing(review, okt, remove_stopwords=False, stop_words=[]):
    # 함수의 인자는 다음과 같다.
    # review : 전처리할 텍스트
    # okt : okt 객체를 반복적으로 생성하지 않고 미리 생성후 인자로 받는다.
    # remove_stopword : 불용어를 제거할지 선택 기본값은 False
    # stop_word : 불용어 사전은 사용자가 직접 입력해야함 기본값은 비어있는 리스트

    # 1. 한글 및 공백을 제외한 문자 모두 제거.
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", review)

    # 2. okt 객체를 활용해서 형태소 단위로 나눈다.
    word_review = okt.morphs(review_text, stem=True)

    if remove_stopwords:
        # 불용어 제거(선택적)
        word_review = [token for token in word_review if not token in stop_words]

    return word_review


# from tqdm import tqdm_notebook
from tqdm import tqdm
# # tqdm은 프로그레스바를 보여주는 유틸

clean_train_review = []
# # tqdm으로 싸주기만 하면되는데, enumerate하면 잘 안됨
for review in tqdm(train_data['document']):
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        clean_train_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_train_review.append([])  #string이 아니면 비어있는 값 추가

test_data = pd.read_csv(data_in_path + 'ratings_test.txt', header=0, delimiter='\t', quoting=3 )

clean_test_review = []

for review in tqdm(test_data['document']):
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        clean_test_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_review.append([])  #string이 아니면 비어있는 값 추가

print('clean_train_review: ', clean_train_review[:3])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_review)
print('Tokenizer Fitting Complete')

train_sequences = tokenizer.texts_to_sequences(clean_train_review)
print('train_sequences: ', clean_train_review[:3], '\n', train_sequences[:3])
test_sequences = tokenizer.texts_to_sequences(clean_test_review)

word_vocab = tokenizer.word_index                                                   # 단어 사전 형태

max_length = 20         # 문장 최대 길이

train_inputs = pad_sequences(train_sequences, maxlen=max_length, padding='post')    # 학습 데이터를 벡터화
train_labels = np.array(train_data['label'])                                        # 학습 데이터의 라벨

test_inputs = pad_sequences(test_sequences, maxlen=max_length, padding='post')      # 테스트 데이터를 벡터화
test_labels = np.array(test_data['label'])                                          # 테스트 데이터의 라벨
#
# train_input_data = 'nsmc_train_input.npy'
# train_label_data = 'nsmc_train_label.npy'
# test_input_data = 'nsmc_test_input.npy'
# test_label_data = 'nsmc_test_label.npy'
# data_configs = 'data_configs.json'
#
# configs = {}
#
# configs['vocab'] = word_vocab
# configs['vocab_size'] = len(word_vocab) + 1 # vocab size 추가

# import os
# # 저장하는 디렉토리가 존재하지 않으면 생성
# if not os.path.exists(data_in_path):
#     os.makedirs(data_in_path)
#
# # 전처리 된 학습 데이터를 넘파이 형태로 저장
# np.save(open(data_in_path + train_input_data, 'wb'), train_inputs)
# np.save(open(data_in_path + train_label_data, 'wb'), train_labels)
# # 전처리 된 테스트 데이터를 넘파이 형태로 저장
# np.save(open(data_in_path + test_input_data, 'wb'), test_inputs)
# np.save(open(data_in_path + test_label_data, 'wb'), test_labels)
#
# # 데이터 사전을 json 형태로 저장
# json.dump(configs, open(data_in_path + data_configs, 'w'), ensure_ascii=False)  # 한글이라서 false
#
# import pickle
#
# clean_train_data = 'clean_train_data.pkl'
# clean_test_data = 'clean_test_data.pkl'
#
# pickle.dump(clean_train_review, open(data_in_path + clean_train_data, 'wb'))
# pickle.dump(clean_test_review, open(data_in_path + clean_test_data, 'wb'))
