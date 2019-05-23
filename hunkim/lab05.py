import tensorflow as tf
import numpy as np
np.set_printoptions(precision=2)

# Logistic Regression 구현
def logistic_regression_basic():
    x_data = [[1, 2],
              [2, 3],
              [3, 1],
              [4, 3],
              [5, 3],
              [6, 2]]
    y_data = [[0], [0], [0], [1], [1], [1]]

    X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    # placeholder 만들때 shape에 주의하자!!

    W = tf.Variable(tf.random_normal([2, 1]), dtype=tf.float32, name='weight')
    b = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='bias')

    hx = tf.sigmoid(tf.matmul(X, W) + b)

    cost = tf.reduce_mean(-Y * tf.log(hx) - (1 - Y) * tf.log(1-hx))

    optimizer = tf.train.GradientDescentOptimizer(0.01)

    train = optimizer.minimize(cost)

    pred = tf.cast(hx > 0.5, dtype=tf.float32)
    '''
    tf.cast : 텐서를 새로운 형태로 캐스팅하는데 사용
    부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
    Boolean형태인 경우 True이면 1, False이면 0을 출력
    '''
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(20000):
            _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
            if step % 1000 == 0:
                print(cost_val)
        pred_result, accuracy_val = sess.run([pred, accuracy], feed_dict={X: x_data, Y: y_data})
        print(pred_result, accuracy_val)


# logistic_regression_basic()

# 당뇨병 데이터를 사용한 예측(diabetes)
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', skiprows=1)
print(xy[:10])
'''
[[-0.88 -0.15  0.08 -0.41  0.   -0.21 -0.77 -0.67  1.  ]
 [-0.06  0.84  0.05  0.    0.   -0.31 -0.49 -0.63  0.  ]
 [-0.88 -0.11  0.08 -0.54 -0.78 -0.16 -0.92  0.    1.  ]
 [ 0.    0.38 -0.34 -0.29 -0.6   0.28  0.89 -0.6   0.  ]
 [-0.41  0.17  0.21  0.    0.   -0.24 -0.89 -0.7   1.  ]
 [-0.65 -0.22 -0.18 -0.35 -0.79 -0.08 -0.85 -0.83  0.  ]
 [ 0.18  0.16  0.    0.    0.    0.05 -0.95 -0.73  1.  ]
 [-0.76  0.98  0.15 -0.09  0.28 -0.09 -0.93  0.07  0.  ]
 [-0.06  0.26  0.57  0.    0.    0.   -0.87  0.1   0.  ]
 [-0.53  0.11  0.51  0.    0.    0.12 -0.9  -0.7   1.  ]]
각 컬럼의 의미는 모른다. 다만 마지막 컬럼은 예측 결과로 보임.
# print(np.shape(xy))     # (758, 9)
shape이 위와 같으므로, 입력 feature 갯수는 8개
  '''

x_data = xy[:, :-1]
# y_data = xy[:, -1]    # y_data의 shape에 영향을 끼치는 차이이므로, 알아두자.
y_data = xy[:, [-1]]    # 이렇게 indexing 해야만, rank2의 tensor를 출력한다!!
print(x_data[:3])
print(y_data[:3])

X = tf.placeholder(dtype=tf.float32, shape=[None, 8])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# placeholder 만들때 shape에 주의하자!!

W = tf.Variable(tf.random_normal([8, 1]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='bias')

hx = tf.sigmoid(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-Y * tf.log(hx) - (1 - Y) * tf.log(1 - hx))

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(cost)

pred = tf.cast(hx > 0.5, dtype=tf.float32)
'''
tf.cast : 텐서를 새로운 형태로 캐스팅하는데 사용
부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
Boolean형태인 경우 True이면 1, False이면 0을 출력
'''
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(30000):
        _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print(cost_val)
    pred_result, accuracy_val = sess.run([pred, accuracy], feed_dict={X: x_data, Y: y_data})
    # print(pred_result, accuracy_val)
    print('accuracy: ', accuracy_val)
