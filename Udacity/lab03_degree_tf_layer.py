import tensorflow as tf
import matplotlib.pyplot as plt
# import keras


# create a model to change the degree of celsius to fahrenheit with tensorflow
# celsius inputs : [-40, -10, 0, 8, 15, 22, 38]
# fahrenheit outputs : [-40, 14, 32, 46, 59, 72, 100]

x = [-40, -10, 0, 8, 15, 22, 38]
y = [-40, 14, 32, 46, 59, 72, 100]

w = tf.Variable(10.0)
b = tf.Variable(10.0)

xx = tf.placeholder(tf.float32)
yy = tf.placeholder(tf.float32)

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
# the units is the neuron, so we must define the number of neurons
# in this case, the neuron has single value weight and single value bias

model = tf.keras.Sequential([l0])
# the codel is composed with only single layer l0
# the code above can be changed to below
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=[1])
# ])


model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
# the model must be compiled with error function and update rule

history = model.fit(x, y, epochs=100000, verbose=False)
# train the model by fit method
# 1 epoch is full iteration of input features

plt.xlabel('Epoch')
plt.ylabel('loss')
plt.plot(history.history['loss'])
plt.savefig('lab03_degree_keras_loss.png')

print(model.predict([100.0]))   # original answer is 212
