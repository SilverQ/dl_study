import tensorflow as tf


def Multi_Variable_linear_regression():
    # Multi Variable linear regression

    # H(x1, x2, x3) = w1x1 + w2x2 + w3x3 + 0 (no bias)

    x1_data = [73., 93., 89., 96., 73.]
    x2_data = [80., 88., 91., 98., 66.]
    x3_data = [75., 93., 90., 100., 70.]
    y_data = [152., 185., 180., 196., 142.]


    W1 = tf.Variable(tf.random_normal([1]), name='Weight1')
    W2 = tf.Variable(tf.random_normal([1]), name='Weight2')
    W3 = tf.Variable(tf.random_normal([1]), name='Weight3')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    xx1 = tf.placeholder(dtype=tf.float32)
    xx2 = tf.placeholder(dtype=tf.float32)
    xx3 = tf.placeholder(dtype=tf.float32)
    yy = tf.placeholder(dtype=tf.float32)

    hx = xx1 * W1 + xx2 * W2 + xx3 * W3 + b
    cost = tf.reduce_mean(tf.square(hx - yy))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
    train = optimizer.minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20):
            feed_dict = {xx1: x1_data, xx2: x2_data, xx3: x3_data, yy: y_data}
            _, cost_val = sess.run([train, cost], feed_dict=feed_dict)
            print(cost_val)
        W1_val, W2_val, W3_val, b_val = sess.run([W1, W2, W3, b])
        print(W1_val, W2_val, W3_val, b_val)


# x의 배열을 행과 열을 바꾸고, 매트릭스 연산으로 시도해보자!!!!
