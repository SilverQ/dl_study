import tensorflow as tf
import numpy as np
np.set_printoptions(precision=2)
tf.set_random_seed(777)  # for reproducibility


# tf.nn.softmax_cross_entropy_with_logits을 알아보자.
def cross_entropy():
    '''
    이전 예제의 코드.
    hx = tf.nn.softmax(tf.matmul(X, W) + b)
    위 문장에서 "tf.matmul(X, W) + b"에 해당하는 부분이 logit이다.
    # cross entropy cost가 아래 코드에서는 -tf.reduce_sum(Y * tf.log(hx), axis=1)으로 표현됨.
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hx), axis=1))

    with_logit의 의미는, hypothesis에 logit을 입력해달라는 것.

    앞으로는 위 표현을 아래와 같이 쓸 수 있다.
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
    cost = tf.reduce_mean(cost_i)

    '''

    x_data = [[1, 2, 1, 1],
              [2, 1, 3, 2],
              [3, 1, 3, 4],
              [4, 1, 5, 5],
              [1, 7, 5, 5],
              [1, 2, 5, 6],
              [1, 6, 6, 6],
              [1, 7, 7, 7]]
    y_data = [[0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [1, 0, 0]]

    X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

    W = tf.Variable(tf.random_normal([4, 3]), dtype=tf.float32, name='weight')
    b = tf.Variable(tf.random_normal([3]), dtype=tf.float32, name='bias')

    logits = tf.matmul(X, W) + b
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    cost = tf.reduce_mean(cost_i)

    '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print('cost_i: ', sess.run(cost_i, feed_dict={X: x_data, Y: y_data}))
    print('cost: ', sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    cost_i:  [ 6.91 11.16 13.9  12.92 20.79 17.79  0.    0.  ]
    cost:  10.434711
    '''
    optimizer = tf.train.GradientDescentOptimizer(0.001)

    train = optimizer.minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(20000):
            _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
            if step % 1000 == 0:
                print(cost_val)
        pred = sess.run(tf.argmax(logits, axis=1), feed_dict={X: x_data, Y: y_data})
        print(pred)


# cross_entropy()


# reshape에 대해 알아보자.
def change_onehot():
    x_data = [[1, 2, 1, 1],
              [2, 1, 3, 2],
              [3, 1, 3, 4],
              [4, 1, 5, 5],
              [1, 7, 5, 5],
              [1, 2, 5, 6],
              [1, 6, 6, 6],
              [1, 7, 7, 7]]
    y_data = [[2], [2], [2], [1], [1], [1], [0], [0]]   # rank=2
    # y_data = [[0, 0, 1],
    #           [0, 0, 1],
    #           [0, 0, 1],
    #           [0, 1, 0],
    #           [0, 1, 0],
    #           [0, 1, 0],
    #           [1, 0, 0],
    #           [1, 0, 0]]
    nb_classes = 3
    # y_data를 one hot coding하지 않은 채로 feed_dict에 입력하고자 하면,
    # Y의 shape은 [None, 3]가 아닌 [None, 1]이 되어야 할 것이며,
    # 그 다음으로는 placeholder객체 Y의 형태를 바꿔주는 작업이 필요하다!
    X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    Y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

    Y_one_hot_1 = tf.one_hot(Y, nb_classes)
    # 문제는 위와 같이 형태를 바꿔줄 경우 shape=[None, 1] -> [None, 1, nb_classes]으로 rank가 하나 증가하게 됨.

    Y_one_hot = tf.reshape(Y_one_hot_1, [-1, nb_classes])


    W = tf.Variable(tf.random_normal([4, 3]), dtype=tf.float32, name='weight')
    b = tf.Variable(tf.random_normal([3]), dtype=tf.float32, name='bias')

    logits = tf.matmul(X, W) + b
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    cost = tf.reduce_mean(cost_i)

    '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print('Y: ', sess.run(Y, feed_dict={X: x_data, Y: y_data}))
    print('Y_one_hot_1: ', sess.run(Y_one_hot_1, feed_dict={X: x_data, Y: y_data}))
    print('Y_one_hot: ', sess.run(Y_one_hot, feed_dict={X: x_data, Y: y_data}))
    # shape을 맞춰줘야 cost 비교할 때 logit과 rank가 동일하게 해줄 수 있다.
    
    # print('cost_i: ', sess.run(cost_i, feed_dict={X: x_data, Y: y_data}))
    # print('cost: ', sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    cost_i:  [ 6.91 11.16 13.9  12.92 20.79 17.79  0.    0.  ]
    cost:  10.434711
    '''
    optimizer = tf.train.GradientDescentOptimizer(0.001)

    train = optimizer.minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(20000):
            _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
            if step % 1000 == 0:
                print(cost_val)
        pred = sess.run(tf.argmax(logits, axis=1), feed_dict={X: x_data, Y: y_data})
        print(pred)


# change_onehot()


# zoo 데이터를 사용한 분류
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
# print(xy[:3])

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# print(x_data[:3])
# print(y_data[:3])
# print(np.shape(x_data))     # (101, 16), shape에 유의하자. 17개 컬럼.

nb_classes = 7                  # range [0",6]

X = tf.placeholder(dtype=tf.float32, shape=[None, 16])
Y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

'''
# https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data,,,,,,,,,,,,,,,,
#   1. animal name:     (deleted),,,,,,,,,,,,,,,,
#   2. hair     Boolean",,,,,,,,,,,,,,,,
#   3. feathers     Boolean",,,,,,,,,,,,,,,,
#   4. eggs     Boolean",,,,,,,,,,,,,,,,
#   5. milk     Boolean",,,,,,,,,,,,,,,,
#   6. airborne     Boolean",,,,,,,,,,,,,,,,
#   7. aquatic      Boolean",,,,,,,,,,,,,,,,
#   8. predator     Boolean",,,,,,,,,,,,,,,,
#   9. toothed      Boolean",,,,,,,,,,,,,,,,
#  10. backbone     Boolean",,,,,,,,,,,,,,,,
#  11. breathes     Boolean",,,,,,,,,,,,,,,,
#  12. venomous     Boolean",,,,,,,,,,,,,,,,
#  13. fins     Boolean",,,,,,,,,,,,,,,,
#  14. legs     Numeric (set of values: {0",2,4,5,6,8}),,,,,,,,,,,
#  15. tail     Boolean",,,,,,,,,,,,,,,,
#  16. domestic     Boolean",,,,,,,,,,,,,,,,
#  17. catsize      Boolean",,,,,,,,,,,,,,,,
#  18. type     Numeric (integer values in range [0",6]),,,,,,,,,,,,,,,
'''

Y_one_hot_1 = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot_1, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot_1)
cost = tf.reduce_mean(cost_i)

pred = tf.argmax(logits, axis=1)
correct_pred = tf.equal(pred, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        _, cost_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(cost_val, acc_val)
    # print(sess.run(correct_pred, feed_dict={X: x_data, Y: y_data}))
    # print(pred)
    # print(sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
    # print(np.reshape(y_data, [-1]))

