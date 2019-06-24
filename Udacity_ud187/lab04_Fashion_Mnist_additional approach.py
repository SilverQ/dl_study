from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Improve progress bar display
import tqdm
import tqdm.auto
# from keras.models import model_from_json
# from keras.callbacks import ModelCheckpoint


tf.logging.set_verbosity(tf.logging.ERROR)
tqdm.tqdm = tqdm.auto.tqdm

print(tf.__version__)       # 1.12.0

tf.enable_eager_execution()

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


def load_model(model_name):
    # Load Model
    loaded_model_json = open(model_name, 'r').read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)

    print('Load Weight Finished')
    return loaded_model


def create_model():
    # Build a model only with fully connected layers
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu,
                               padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu, padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model


if 1 == 1:
    model = load_model('Fashion_Mnist.json')
else:
    model = create_model()
    # Save the model
    model_json = model.to_json()
    with open('Fashion_Mnist.json', 'w') as json_file:
        json_file.write(model_json)


model.summary()

# Compile the model
opt = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
batch_size = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# action = 'Train'
action = 'Eval'

if action == 'Train':
    model.fit(train_dataset, epochs=1, steps_per_epoch=math.ceil(num_train_examples/batch_size))
    # Save the weights
    model.save_weights('Fashion_Mnist.h5')
    print('Save model Finished')
elif action == 'Eval':
    model.load_weights('Fashion_Mnist.h5')
    print('Load model Finished')


# Evaluate accuracy
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/batch_size))
print('Accuracy on testset: {}'.format(test_accuracy))

# Make predictions and explore
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
print(predictions.shape)
print(predictions[0])
'''
(32, 10)
[9.9332246e-04 3.0070751e-07 5.4520708e-02 5.4992433e-03 5.0916857e-01
 4.4711786e-08 4.2935511e-01 2.3336844e-08 4.6259694e-04 3.6422879e-08]
'''

print(np.argmax(predictions[0]))
print(test_labels[0])


def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_labels, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_labels:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:0.2f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_labels]), color=color)


def plot_value_array(i, predictions_array, true_labels):
    predictions_array, true_labels = predictions_array[i], true_labels[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_labels].set_color('blue')


# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, test_labels)
# plt.savefig('FashionMnist.png')

# num_rows = 5
# num_cols = 3
# num_images = num_rows * num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, predictions, test_labels)
# plt.savefig('FashionMnist.png')

img = test_images[0]
print(img.shape)

img = np.array([img])
print(img.shape)

prediction_single = model.predict(img)
print(prediction_single)

plot_value_array(0, prediction_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(prediction_single[0])
