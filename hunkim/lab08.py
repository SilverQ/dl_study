import numpy as np
import tensorflow as tf


def rank1():
    a = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    print(a.ndim)       # 1
    print(a.shape)      # (10,) : 10 뒤의 쉼표에 유의하자.


def rank2():
    b = np.array([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.],
                  [10., 11., 12.]])
    print(b.ndim, b.shape)  # 2, (4, 3) : 2차원, 1st element 4개, 각각 3개 element로 구성됨.


def tensor_rank():
    c = tf.constant([1, 2, 3, 4])
    d = tf.constant([[1, 2],
                     [3, 4]])
    with tf.Session() as sess:
        print(tf.shape(c).eval())   # [4]
        print(tf.shape(d).eval())   # [2 2], numpy에서 출력하는 형태는 (2, 2)였을 것. 차이를 기억해두자. 계속 헷갈린다.

        e = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                          [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
        print(e)    # Tensor("Const_2:0", shape=(1, 2, 3, 4), dtype=int32)
        # e라는 객체를 출력할 때는, shape을 numpy에서 표현하는 방식으로 보여준다.
        print(e.eval())     # axis 숫자는 가장 바깥쪽 element에 0, 가장 안쪽 elements에 마지막 숫자가 붙는다.


def tensor_broadcast():
    with tf.Session():
        mat1 = tf.constant([1, 2])
        con1 = tf.constant(2)
        # print(tf.add(mat1, con1).eval())    # [3 4]

        mat2 = tf.constant([3, 4])
        # print(tf.add(mat1, mat2).eval())    # [4 6] : shape이 동일하여 element wise sum

        mat3 = tf.constant([[3], [4]])
        print(tf.add(mat1, mat3).eval())    # [[4 5]
                                            #  [5 6]] : [1, 2]에 3이 한번 더해지고, 4도 한번 더해져 (2, 2) matrix가 됨
        # broadcasting으로 인해 생각지 못한 값이 많이 만들어진다. 유의해서 사용하자.


def reduced_sum():
    with tf.Session():
        print(tf.reduce_mean([1, 2]).eval())    # 1
        print(tf.reduce_mean([1., 2.]).eval())    # 1.5, 정수로 입력하면 정수로 출력되는데, 반올림이 아니라 버림?

        x = [[1, 2],
             [3, 4]]
        x = np.float32(x)
        print(tf.reduce_sum(x, axis=0).eval())  # [4., 6.]
        print(tf.reduce_sum(x, axis=1).eval())  # [3., 7.], 문제는 어느 축으로 더하든, axis로 사용된 축의 rank가 하나 감소한 결과
        print(tf.reduce_sum(x, axis=-1).eval())  # [3., 7.], -1은 가장 안쪽의 dimension에서 합산
        print(tf.reduce_sum(x).eval())  # 10, 축을 입력하지 않으면 모든 elements를 합


def argmax_test():
    with tf.Session():
        x = tf.constant([[0, 1, 2],
                         [4, 5, 6]])
        print(tf.argmax(x).eval())  # [1 1 1], 축을 입력하지 않으면 axis=0
        print(tf.argmax(x, axis=0).eval())  # [1 1 1]
        print(tf.argmax(x, axis=1).eval())  # [2 2]


def reshape_test():
    t = np.array(([[[0, 1, 2],
                    [3, 4, 5]],
                   [[6, 7, 8],
                    [9, 10, 11]]]))
    print(t.shape)      # (2, 2, 3)

    print(np.reshape(t, [-1, 3]))
    with tf.Session():
        print(tf.reshape(t, [-1, 3]).eval())
        print(tf.reshape(t, [-1, 1, 3]).eval())
        print(tf.squeeze(t).eval())
    '''
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    
    [[[ 0  1  2]]
    
     [[ 3  4  5]]
    
     [[ 6  7  8]]
    
     [[ 9 10 11]]]
     
    [[[ 0  1  2]
      [ 3  4  5]]
    
     [[ 6  7  8]
      [ 9 10 11]]]  # 이건 모양이 같은거 아닌가? 좀더 파보자
    '''


# reshape_test()

# https://m.blog.naver.com/wideeyed/221164750517
# 여기의 설명은 "차원 중 크기가 1인 것을 찾아 제거

def reshape_advance():
    t = np.array(([[[0, 1, 2]],
                   [[3, 4, 5]],
                   [[6, 7, 8]],
                   [[9, 10, 11]]]))
    print(t.shape)  # (4, 1, 3)
    with tf.Session() as sess:
        t1 = tf.squeeze(t)
        print(t1)       # Tensor("Squeeze:0", shape=(4, 3), dtype=int64)
        print(sess.run(t1))
        # print(sess.run(t1.shape))

        a = [0, 1, 2, 3]
        b = tf.expand_dims(a, axis=0)
        c = tf.expand_dims(a, axis=1)
        print(sess.run(b))      # [[0 1 2 3]]
        print(sess.run(c))
        '''
        [[0]
         [1]
         [2]
         [3]]
        '''
        d = tf.one_hot([0, 1, 2, 3, 4, 5], depth=0)     # 아무 정보도 표현하지 못함
        e = tf.one_hot([0, 1, 2, 3, 4, 5], depth=1)     # 1개 element로 표현
        f = tf.one_hot([0, 1, 2, 3, 4, 5], depth=2)     # 2개 element로 표현
        g = tf.one_hot([0, 1, 2, 3, 4, 5], depth=3)
        print(sess.run(d))  # []
        print(sess.run(e))
        '''
        [[1.]
         [0.]
         [0.]
         [0.]
         [0.]
         [0.]]
        '''
        print(sess.run(f))
        '''
        [[1. 0.]
         [0. 1.]
         [0. 0.]
         [0. 0.]
         [0. 0.]
         [0. 0.]]
        '''
        print(sess.run(g))
        '''
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]
         [0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
        '''
        h = tf.one_hot([0, 1, 2, 3, 4, 5], depth=6)     # 6개 element로 one hot 표현
        print(sess.run(h))
        '''
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
        '''

        a = tf.cast([1.2, 3.2, 4., 5.5], tf.int32)
        print(sess.run(a))      # [1 3 4 5]
        b = tf.cast([True, False, 1==1, 1==2], tf.int32)
        print(sess.run(b))      # [1 0 1 0]


# reshape_advance()

# stack

x = [1, 2]
y = [3, 4]
z = [5, 6]

d = tf.stack([x, y, z])
e = tf.stack([x, y, z], axis=0)
f = tf.stack([x, y, z], axis=1)
# g = tf.stack([x, y, z], axis=2)   # error@!!
sess = tf.Session()
print(sess.run(d))
print(sess.run(e))
print(sess.run(f))
# print(sess.run(g))
'''
[[1 2]
 [3 4]
 [5 6]]

[[1 2]
 [3 4]
 [5 6]]

[[1 3 5]
 [2 4 6]]
'''

e1 = tf.ones_like(e)
f1 = tf.zeros_like(f)
print(sess.run(e1))
print(sess.run(f1))
'''
[[1 1]
 [1 1]
 [1 1]]
[[0 0 0]
 [0 0 0]]
'''

for x, y in zip([1, 2, 3], [4, 5, 6]):      # [1, 2, 3]이 x, [4, 5, 6]이 y, x와 y에 대해 3번 실행.
    print(x)        # 1 2 3
    print(y)        # 4 5 6
    print(x, y)     # 1 4   2 5     3 6

# for x, y in zip([1, 2], 4]):    # error!!!!!
#     print(x)
#     print(y)
