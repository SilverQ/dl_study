import tensorflow as tf
import matplotlib.pyplot as plt


def cost_chart():
    # Cost가 W에 관한 함수 임을 관찰하자.
    x = [1, 2, 3]
    y = [1, 2, 3]

    xx = tf.placeholder(dtype=tf.float32)
    WW = tf.placeholder(dtype=tf.float32)
    # W 역시 업데이트 후 다시 입력해주기 위해 placeholder로 설정

    hx = xx * WW
    cost = tf.reduce_mean(tf.square(hx - y))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        w_val = []  # 차트를 그리기 위해 x, y값을 수집하는 array
        cost_val = []

        for i in range(-10, 31):
            cost_val.append(sess.run([cost], feed_dict={xx: x, WW: i*0.1}))
            w_val.append(i*0.1)
        plt.plot(w_val, cost_val)
        plt.show()


# gradient를 계산해보고 직접 W를 업데이트해보자.
x = [1, 2, 3]
y = [1, 2, 3]
W = tf.Variable(initial_value=-7.0)

xx = tf.placeholder(dtype=tf.float32)

hx = xx * W
cost = tf.reduce_mean(tf.square(hx - y))
grad = tf.reduce_mean((hx - y) * xx * 0.1)
grad_update = W.assign_sub(grad)     # gradient를 차감해줌.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30):
        W_val = sess.run(W)
        _, cost_val, grad_val = sess.run([grad_update, cost, grad], feed_dict={xx: x})
        print(W_val, grad_val)
