from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
#
# for cl in classes:
#     img_path = os.path.join(base_dir, cl)
#     images = glob.glob(img_path + '/*.jpg')
#     print("{}: {} Images".format(cl, len(images)))
#     train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]
#
#     for t in train:
#         if not os.path.exists(os.path.join(base_dir, 'train', cl)):
#             os.makedirs(os.path.join(base_dir, 'train', cl))
#         shutil.move(t, os.path.join(base_dir, 'train', cl))
#
#     for v in val:
#         if not os.path.exists(os.path.join(base_dir, 'val', cl)):
#             os.makedirs(os.path.join(base_dir, 'val', cl))
#         shutil.move(v, os.path.join(base_dir, 'val', cl))

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

batch_size = 64
IMG_SHAPE = 150


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


image_gen_train = ImageDataGenerator(rescale=1./255,
                                     rotation_range=45,
                                     width_shift_range=0.15,
                                     height_shift_range=0.15,
                                     zoom_range=0.5,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 shuffle=True,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE))

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(5, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
# # train_dir = os.path.join(base_dir, 'train')
# # val_dir = os.path.join(base_dir, 'val')
# # total_train = len(os.listdir(train_dir))
# # total_val = len(os.listdir(val_dir))
#
# total_train = 0
# total_val = 0
# for cl in classes:
#     train_img_path = os.path.join(base_dir, 'train', cl)
#     val_img_path = os.path.join(base_dir, 'val', cl)
#     total_train += len(os.listdir(train_img_path))
#     total_val += len(os.listdir(val_img_path))
#
# print(total_train)
# print(total_val)
# print(int(np.ceil(total_train / float(batch_size))))
# print(int(np.ceil(total_val / float(batch_size))))
#
# EPOCHS = 100
#
# history = model.fit_generator(train_data_gen,
#                               steps_per_epoch=int(np.ceil(total_train / float(batch_size))),
#                               epochs = EPOCHS,
#                               validation_data=val_data_gen,
#                               validation_steps=int(np.ceil(total_val / float(batch_size)))
# )
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(EPOCHS)
#
# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
#
# plt.savefig('./exercise_flowers.png')
# plt.show()

'''
# TODO: Experiment with Different Parameters
So far you've created a CNN with 3 convolutional layers and followed by a fully connected layer with 512 units.
In the cells below create a new CNN with a different architecture.
Feel free to experiment by changing as many parameters as you like.
For example, you can add more convolutional layers, or more fully connected layers.
You can also experiment with
different filter sizes in your convolutional layers,
different number of units in your fully connected layers,
different dropout rates, etc...
You can also experiment by performing image aumentation with more image transformations that we have seen so far.
Take a look at the  [ImageDataGenerator Documentation] to see a full list of all the available image transformations.
(https://keras.io/preprocessing/image/)
For example, you can add shear transformations, or you can vary the brightness of the images, etc...
Experiement as much as you can and compare the accuracy of your various models.
Which parameters give you the best result?
'''

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.summary()

# OPTIMIZER = tf.keras.optimizers.Adam(lr=0.001)
OPTIMIZER = tf.keras.optimizers.Adadelta()
# OPTIMIZER = tf.keras.optimizers.Adagrad()
# OPTIMIZER = tf.keras.optimizers.RMSprop()
# OPTIMIZER = tf.keras.optimizers.SGD

model.compile(optimizer=OPTIMIZER,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

total_train = 0
total_val = 0
for cl in classes:
    train_img_path = os.path.join(base_dir, 'train', cl)
    val_img_path = os.path.join(base_dir, 'val', cl)
    total_train += len(os.listdir(train_img_path))
    total_val += len(os.listdir(val_img_path))

print(total_train)
print(total_val)
print(int(np.ceil(total_train / float(batch_size))))
print(int(np.ceil(total_val / float(batch_size))))

EPOCHS = 100

history = model.fit_generator(train_data_gen,
                              steps_per_epoch=int(np.ceil(total_train / float(batch_size))),
                              epochs = EPOCHS,
                              validation_data=val_data_gen,
                              validation_steps=int(np.ceil(total_val / float(batch_size)))
)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('./exercise_flowers.png')
plt.show()
