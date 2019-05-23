import tensorflow as tf
import numpy as np

# tensorboard를 사용해보자!
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [0, 1, 1, 0]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.reshape(y_data, newshape=[4, 1])
X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

with tf.name_scope('layer1') as scope:
    W1 = tf.Variable(initial_value=tf.random_normal([2, 2]), dtype=tf.float32, name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), dtype=tf.float32, name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    w1_hist = tf.summary.histogram('weight1', W1)  # select tensor to log
    l1_hist = tf.summary.histogram('layer1', layer1)  # select tensor to log
with tf.name_scope('layer2') as scope:
    W2 = tf.Variable(initial_value=tf.random_normal([2, 1]), dtype=tf.float32, name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='bias2')
    hx = tf.sigmoid(tf.matmul(layer1, W2) + b2)

cost = - tf.reduce_mean(Y * tf.log(hx) + (1 - Y) * tf.log(1 - hx))  # 김성훈 교수님 강의
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
pred = tf.cast(hx > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(pred, Y), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# w1_hist = tf.summary.histogram('weight1', W1)  # select tensor to log
cost_sum = tf.summary.scalar('cost', cost)  # select tensor to log
summary = tf.summary.merge_all()  # merge all summary
writer = tf.summary.FileWriter('./logs')  # create writer, 이 타이밍에 파일 생성
# writer.add_graph(sess.graph)

for step in range(501):
    # _, cost_val, acc_val = sess.run([train, cost, acc], feed_dict={X: x_data, Y: y_data})
    # w1_hist = tf.summary.histogram('weight1', W1)   # select tensor to log
    # cost_sum = tf.summary.scalar('cost', cost)      # select tensor to log
    # summary = tf.summary.merge_all()                # merge all summary
    # writer = tf.summary.FileWriter('./logs')        # create writer를 반복문 안에서 돌리지 말자.
    writer.add_graph(sess.graph)                        # 그래프 추가
    s, cost_val, _ = sess.run([summary, cost, train], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, cost_val)
        writer.add_summary(s, global_step=step)     # summary 추가

W1_val, b1_val, W2_val, b2_val = sess.run([W1, b1, W2, b2])
print(W1_val)
print(b1_val)
print(W2_val)
print(b2_val)
# print(acc_val)
