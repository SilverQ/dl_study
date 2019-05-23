import numpy as np

# Intro to python and colab V1
# https://www.youtube.com/watch?v=xp7DGVGf8_c#action=share


# Loop and Break
def loop_with_break():
    while True:
        print('True')
        break


def numpy_basic():
    b = np.array([0, 1, 2, 3, 4, 5])
    print('Max: {}'.format(np.max(b)))
    print('Min: {}'.format(np.min(b)))
    print('Average: {}'.format(np.average(b)))
    print('Max Index: {}'.format(np.argmax(b)))
    '''
    Max: 5
    Min: 0
    Average: 2.5
    Max Index: 5
    
    Process finished with exit code 0
    '''


def random_number():
    c = np.random.rand(3, 3)    # (3, 3) 크기의 난수 array 생성
    print(c, '\n', c.shape)
    '''
    [[0.80735481 0.44091246 0.84936025]
     [0.30170273 0.69716532 0.62856908]
     [0.65515467 0.00877733 0.90497712]]
     (3, 3)
    Process finished with exit code 0
    '''


# shape of array
b = np.array([0, 1, 2, 3, 4, 5])
# print(b.shape)
# (6,)
# 이 shape을 기억해야 한다. 내가 이걸 이 shape으로부터 데이터의 모양을 다시 추측하기 어려워하고 있다.

# Google Colab에서는 커맨드라인 명령을 지원한다
'''
!ls -l
!pwd
!pip install numpy
이와 같이 !를 붙여셔 colab 노트북에서 실행하면 됨
# 현재 디렉토리를 리스트하고, 하위 디렉토리로 이동하여 콘텐트를 리스트해보자
!pwd
!cd /~~~
!ls -l
'''


# Introduction To ML V2
# 섭씨(Celsius)를 화씨(Fahrenheit)로 바꾸는 문제가 있을 때,
# 전통적인 프로그래밍에서는?
def fahren(c):
    return c * 1.8 + 32


# print(fahren(10))
# 50
# 입력으로 c를 받고, 1.8과 32를 사용해서 결과를 계산할 수 있다.
# 위의 식을 알고리즘, c는 입력, f는 출력
# 반면 기계학습은? 1.8과 32가 주어지지 않는다. 다양한 실제 사례(입력과 출력)만 존재할 뿐이다.
# 기계학습은 이처럼 데이터가 주어지고 입력과 출력 사이에 어떤 관계가 존재할지 추론하는 알고리즘(함수)을 탐색하는 역할을 한다.

# inputs : [-40, -10, 0, 8, 15, 22, 38]
# outputs : [-40, 14, 32, 46, 59, 72, 100]

