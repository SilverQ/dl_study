
# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from functions import create_samples

n_features = 2
n_clusters = 3
n_samples_per_cluster = 10
seed = 5
embiggen_factor = 5

np.random.seed(seed)

centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

model = tf.global_variables_initializer()
with tf.Session() as session:
    sample_values = session.run(samples)
    centroid_values = session.run(centroids)

from functions import plot_clusters

plot_clusters(sample_values, centroid_values, n_samples_per_cluster)

plt.show()
