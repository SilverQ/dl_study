from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

print(tf.__version__)       # 1.12.0

# tf.enable_eager_execution()

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

'''
Dataset fashion_mnist downloaded and prepared to /home/hdh/tensorflow_datasets/fashion_mnist/1.0.0. Subsequent calls will reuse this data.
Process finished with exit code 0
'''

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explorer the data
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

print('Number of training examples: {}'.format(num_train_examples))
print('Number of testing examples: {}'.format(num_test_examples))


# Preprocess the data(normalize)
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

'''
# Explorer the progressed data
# Take a single image and remove the color dimension
for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28, 28))

plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()
'''

# Build a model only with fully connected layers
l0 = tf.keras.layers.Flatten(input_shape=(28, 28, 1))       # input
l1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)      # hidden
l2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)    # output

model = tf.keras.Sequential([l0, l1, l2])
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# adam 옵티마이저와 tf.enable_eager_execution()과는 문제가 있는듯.
# https://github.com/tensorflow/tensorflow/issues/25324
# 위 문장을 주석처리하면 정상적으로 실행됨

# train the model
batch_size = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/batch_size))

# Evaluate accuracy
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/batch_size))
print('Accuracy on testset: {}'.format(test_accuracy))

# Make predictions and explore
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
predictions.shape
predictions[0]

# https://classroom.udacity.com/courses/ud187/lessons/e52f6e56-2fbc-4ba8-9f74-377937b7da5c/concepts/69c86393-9b29-4875-b0b6-68b1d3cef4ed
# 05:35 진행. eager 옵션을 주석처리해서 에러 발생.
'''
Traceback (most recent call last):
  File "/home/hdh/Documents/TensorflowStudy/DeepLearning Basic/Udacity/lab04_Fashion_Mnist.py", line 109, in <module>
    for test_images, test_labels in test_dataset.take(1):
  File "/home/hdh/Env/p3ten10/lib/python3.5/site-packages/tensorflow/python/data/ops/dataset_ops.py", line 167, in __iter__
    raise RuntimeError("dataset.__iter__() is only supported when eager "
RuntimeError: dataset.__iter__() is only supported when eager execution is enabled.
'''