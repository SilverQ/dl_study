import tensorflow as tf
import numpy as np


def train_from_file():
    # 파일로부터 데이터를 읽어와서 학습시켜보자.
    xy = np.loadtxt('data-01-test-score.csv', delimiter=',', skiprows=1)

    x_data = xy[:, :-1]
    y_data = xy[:, -1]
    print(x_data[:3])

    y_data = np.reshape(y_data, (-1, 1))
    print(y_data[:3])

    W = tf.Variable(tf.random_normal([3, 1]), name='Weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # xx = tf.placeholder(dtype=tf.float32)
    xx = tf.placeholder(dtype=tf.float32, shape=[None, 3])

    hx = tf.matmul(xx, W) + b
    # (~, 3) x (3, 1)

    cost = tf.reduce_mean(tf.square(hx - y_data))

    optimizer = tf.train.GradientDescentOptimizer(0.00001)

    train = optimizer.minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5):
            feed_dict = {xx: x_data}
            _, cost_val = sess.run([train, cost], feed_dict=feed_dict)
            print(cost_val)
        W_val, b_val = sess.run([W, b])
        print(W_val, b_val)


# train_from_file()


def train_from_file_with_queue():
    # 여러 파일을 읽어오고 queue를 사용해서 배치 학습을 시켜보자.
    files = ['data-02-test-score.csv']

    filename_queue = tf.train.string_input_producer(files, shuffle=False, name='filename_queue')

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    record_defaults = [[0.], [0.], [0.], [0.]]
    xy = tf.decode_csv(value, record_defaults=record_defaults)

    train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=2)

    X = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3, 1]), name='Weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hx = tf.matmul(X, W) + b

    cost = tf.reduce_mean(tf.square(hx - Y))

    optimizer = tf.train.GradientDescentOptimizer(0.00001)

    train = optimizer.minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(2):
            x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
            feed_dict = {X: x_batch, Y: y_batch}
            _, cost_val = sess.run([train, cost], feed_dict=feed_dict)
            print(cost_val)
        # W_val, b_val = sess.run([W, b])
        # print(W_val, b_val)

        # 남아있는 궁금증. queue를 사용해서 연산할 때 위 예제에서 batch size=2로 설정했음.
        # 그렇다면 csv 파일 내의 6개 instance를 3개씩 나눠서 배치를 실행했을지. 그렇다면 cost는 3번 실행할 때의 cost의 합? 평균?
        # 참 작게 시작했는데, 왜 그럴깡?


train_from_file()
train_from_file_with_queue()

'''
183874.05
67728.31
24956.947
9206.127
3405.7861
[[ 0.35959032]
 [-0.50381863]
 [ 1.6976942 ]] [-0.62469536]


37517.285
15061.585

'''