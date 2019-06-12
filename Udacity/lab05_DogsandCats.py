from __future__ import absolute_import, division
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras_preprocessing.image import ImageDataGenerator
tf.logging.set_verbosity(tf.logging.ERROR)
tf.enable_eager_execution()

# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)

# 참고
# https://www.tensorflow.org/datasets

'''
print(tfds.list_builders())
[
 'abstract_reasoning', 'bair_robot_pushing_small', 'caltech101', 'cats_vs_dogs',
 'celeb_a', 'celeb_a_hq', 'chexpert', 'cifar10', 'cifar100', 'cifar10_corrupted', 
 'cnn_dailymail', 'coco2014', 'colorectal_histology', 'colorectal_histology_large', 
 'cycle_gan', 'diabetic_retinopathy_detection', 'dsprites', 'dtd', 'dummy_dataset_shared_generator', 
 'dummy_mnist', 'emnist', 'fashion_mnist', 'flores', 'glue', 'groove', 'higgs', 
 'horses_or_humans', 'image_label_folder', 'imagenet2012', 'imagenet2012_corrupted', 'imdb_reviews', 
 'iris', 'kmnist', 'lm1b', 'lsun', 'mnist', 'moving_mnist', 'multi_nli', 'nsynth', 'omniglot', 
 'open_images_v4', 'oxford_flowers102', 'oxford_iiit_pet', 'para_crawl', 'quickdraw_bitmap', 
 'rock_paper_scissors', 'shapes3d', 'smallnorb', 'squad', 'starcraft_video', 'sun397', 'svhn_cropped', 
 'ted_hrlr_translate', 'ted_multi_translate', 'tf_flowers', 'titanic', 'ucf101', 'voc2007', 
 'wikipedia', 'wmt15_translate', 'wmt16_translate', 'wmt17_translate', 'wmt18_translate', 
 'wmt19_translate', 'wmt_translate', 'xnli']
'''

dataset, metadata = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True, split=tfds.Split.TRAIN)
dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

'''
print('metadata: ', metadata)
tfds.core.DatasetInfo(
    name='cats_vs_dogs',
    version=2.0.1,
    description='A large set of images of cats and dogs.There are 1738 corrupted images that are dropped.',
    urls=['https://www.microsoft.com/en-us/download/details.aspx?id=54765'],
    features=FeaturesDict({
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'image/filename': Text(shape=(), dtype=tf.string, encoder=None),
        'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=2)
    },
    total_num_examples=23262,
    splits={
        'train': <tfds.core.SplitInfo num_examples=23262>
    },
    supervised_keys=('image', 'label'),
    citation='"""
        @Inproceedings (Conference){asirra-a-captcha-that-exploits-interest-aligned-manual-image-categorization,
        author = {Elson, Jeremy and Douceur, John (JD) and Howell, Jon and Saul, Jared},
        title = {Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization},
        booktitle = {Proceedings of 14th ACM Conference on Computer and Communications Security (CCS)},
        year = {2007},
        month = {October},
        publisher = {Association for Computing Machinery, Inc.},
        url = {https://www.microsoft.com/en-us/research/publication/asirra-a-captcha-that-exploits-interest-aligned-manual-image-categorization/},
        edition = {Proceedings of 14th ACM Conference on Computer and Communications Security (CCS)},
        }
        
    """',
    redistribution_info=,
)
'''

print(metadata.splits['train'].num_examples)    # 23262
print(metadata.features['label'].num_classes)    # 2

# https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3


# for features in dataset.take(1):
#     image, label = features['image'], features['label']
#
# print(dataset.keys())   # dict_keys(['train'])
# print(dataset)          # {'train': <DatasetV1Adapter shapes: ((?, ?, 3), ()), types: (tf.uint8, tf.int64)>}


# train_dataset, test_dataset = dataset['train'], dataset['test']
#
#
# num_train_examples = metadata.splits['train'].num_examples
# num_test_examples = metadata.splits['test'].num_examples
#
# print('Number of training examples: {}'.format(num_train_examples))
# print('Number of testing examples: {}'.format(num_test_examples))

# https://classroom.udacity.com/courses/ud187/lessons/1771027d-8685-496f-8891-d7786efb71e1/concepts/8b8c3d93-4117-4134-b678-77d54634b656
# 0:52 진행 중
