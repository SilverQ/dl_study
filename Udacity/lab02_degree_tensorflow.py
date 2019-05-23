import tensorflow as tf


# create a model to change the degree of celsius to fahrenheit with tensorflow
# celsius inputs : [-40, -10, 0, 8, 15, 22, 38]
# fahrenheit outputs : [-40, 14, 32, 46, 59, 72, 100]

x = [-40, -10, 0, 8, 15, 22, 38]
y = [-40, 14, 32, 46, 59, 72, 100]

w = tf.Variable(10.0)
b = tf.Variable(10.0)

xx = tf.placeholder(tf.float32)
yy = tf.placeholder(tf.float32)

hx = w * xx + b

cost = tf.reduce_mean(tf.square(yy - hx))

learning_rate = 0.1
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(10001):
        [_, cost_val, w_val, b_val] = sess.run([train, cost, w, b], feed_dict={xx: x, yy: y})
        if i % 1000 == 0:
            print(cost_val, w_val, b_val)

print('The function for change degree from Celsius to Fahrenheit:'+'\n'+'F = {} x C + {}'.format(w_val, b_val))
