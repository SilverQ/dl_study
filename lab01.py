import tensorflow as tf
import numpy as np


def helloworld():
    hello = tf.constant("Hello World")

    print(hello)
    # Tensor("Const:0", shape=(), dtype=string)
    # tf.constant를 사용해 상수 노드 생성, 이는 아직 computational graph에 속함
    # 보통의 프로그램에서는 안에 있는 값을 바로 볼 수 있지만,
    # tensorflow에서는 session을 열어서 실행해야 함

    sess = tf.Session()
    # Computational Graph 생성

    # sess.run(hello)
    # hello 객체가 실제 생성됨
    print(sess.run(hello))
    # b'Hello World' : b가 의미하는 것은 Byte 문자열이라는 뜻


def exer_add():
    # tf를 사용해 덧셈을 해보자.
    a = tf.constant(3.0)
    b = tf.constant(10.0)

    c = a + b
    d = tf.add(a, b)

    sess = tf.Session()

    c_val, d_val = sess.run([c, d])
    # sess.run 안에 여러 변수를 넣을 때 list 형태로 만드는 것을 주의하자.

    print(c_val, d_val)


def using_placeholder():
    # placeholder를 사용해보자
    a = tf.placeholder(dtype=tf.float32)
    b = tf.placeholder(dtype=tf.float32)

    c = a + b
    d = tf.add(a, b)

    sess = tf.Session()

    c_val, d_val = sess.run([c, d], feed_dict={a: 1, b: 2})
    print(c_val, d_val)


# tensor의 개념
"""
3 : a rank 0 tensor!!, 이것은 shape []의 scalar이다.
[1., 2.] : a rank 1 tensor, 이것은 shape [2]의 vector.
[[1., 2.], [1., 3.]] : a rank 2 tensor. 이것은 shape [2, 2]의 matrix
[[[1, 2], [2, 3]], [[1, 2], [2, 3]], [[2, 3], [3, 4]]]
    : a rank 3 tensor, shape (3, 2, 2). 아래처럼 줄바꿈하며 보는 것이 이해가 편하다. 
"""

a = np.array([[1, 2, 3], [2, 3, 4]])
b = np.array([])
c = np.array([[[1, 2], [2, 3]], [[1, 2], [2, 3]], [[2, 3], [3, 4]]])
d = np.array([[[1, 2], [2, 3]],
              [[1, 2], [2, 3]],
              [[2, 3], [3, 4]]])
print(a.shape)  # (2, 3)
print(b.shape)  # (0,)
print(c.shape)  # (3, 2, 2)

# tensor를 논할때는 rank, shape, type을 언급, 이 개념을 꼭 이해하고 넘어가야 함
