"""
# 뉴럴넷
DNN, Fully forward, 등
CNN, RNN,

언어를 어떻게 벡터로 나타낼 것인지가 중요

Text Representation에는 word, character, sentence, document 단위로 나타내는 방법
한글은 자모 단위로 잘라서 임베딩하는 경우도...
각각의 장단점이 있다보니,

예제 ; 나는 학교에 간다, 나는 매일 학교에 늦는다.
one-hot 절차
1. vocabulary 생성
 - 나는:0, 학교에:1, 간다:2, 매일:3, 늦는다:4
2. one-hot
 - 나는: [00001], 학교에 [00010], 간다 [00100], 매일 [01000] 늦는다 [10000]

Bag-of-Words
나는 학교에 간다 [00111], 나는 매일 학교에 늦는다 [11011]
벡터가 단어 혹은 문장의 의미를 담고 있지 않다.

모든 단어의 내적 = 0, orthogonal은 수직, 전혀 연관이 없는 벡터


그렇다면?
Neural word representation
 - Word2Vec, GloVe, Fasttext

Word2Vec - 주변단어를 알면 해당 단어의 뜻을 알 수 이싿
Key Idea - Distribution
Efficient Representation of Word Representations in Vector Space
이걸 완전히 이해하려면 다른 논문 하나를 더 읽어야 함
Efficient Representation of Word and phrases Representations

Cross Entropy : multi class
Binary Cross Entropy : 0 혹은 1 예측
y log y + (1-y) log (1-y)

-log p가 정보량, 정보량과 실제값의 확률분포를 맞게 하도록하는 함수
wevi에서 살펴보자

word2vec의 단점
softmax의 계산량 exp(u v) / sum(exp(u v))
hierarchical softmax : 바이너리 트리로 나타내고, N 번의 계산을 log N 번으로 저감
negative sampling이 연산량도 더 줄이는데 성능도 더 높아
개념은 분모의 갯수를 줄이기 위해 네거티브 관계에 있는 단어들로 골라서...

GloVe는 직접 학습하는 도구가 없고, 학습된 정보만 제공됨
논문을 설명하는 흐름이 다르다. 딥러닝 관점이 아님. 컨셉은 co-occurence matrix를 사용한 통계적 정보 사용
https://www.google.com/search?q=glove+co+occurrence+matrix&source=lnms&tbm=isch&sa=X&ved=0ahUKEwi50-zyi6zkAhU2zIsBHZMXDNYQ_AUIESgB&biw=1064&bih=975

fasttext는 모델 자체는 word2vec이랑 똑같은데, 결정적인 차이는
sub-word를 이용해 오타, 유사어의 의미를 잡아냄
school을 하나의 벡터로 표현했다면, 패스트 텍스트는 sch, cho, hoo, ool의 sub0-word로 나탄고 이들 벡터를 합계
한글은 결국 자모를 분리해서 fasttxt를 적용하는게 좋다더라


edwith 추천강의
신경망과 딥러닝
머신러닝 프로젝트 구조화하기
성능 향상 시키기
bayesian deep learning(최성준) - 예측에 대한 불안정성, 신뢰성을 연구하는 분야
심화된 내용임
논문으로 짚어보면 딥러닝의 맥
주재걸 선형대수
모두를 위한 선형대수학
https://www.edwith.org/search/index?categoryId=71

송영숙 깃헙 젠심 데이터
github.com/songys/Chatbot_data

"""
