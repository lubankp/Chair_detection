import os
import cv2
import json

from matplotlib import pyplot as plt
from augmentation import augmentation_fun
from move_json_to_folders import move_json_to_folders_fun
from init import init

import tensorflow as tf
from load_to_tensorflow import load_aug_data
from prepare_labels import train_labels_fun
from deep_learning import deep_learning_fun
from live_detection import live_detection
from keras.models import load_model

# # --------------------------------------------------------------------------
# # Init
# init()
#
# # --------------------------------------------------------------------------
# # Move .json to folders
# move_json_to_folders_fun()
#
# # --------------------------------------------------------------------------
# Augmentation
# augmentation_fun()

#Load Augumented Data to Tensorflow
data_images = load_aug_data()

# Prepare Labels
data_labels = train_labels_fun()

train = tf.data.Dataset.zip((data_images[0], data_labels[0]))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((data_images[1], data_labels[1]))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((data_images[2], data_labels[2]))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

# data_samples = train.as_numpy_iterator()
# res = data_samples.next()
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx in range(4):
#     sample_image = res[0][idx]
#     sample_coords = res[1][1][idx]
#
#     cv2.rectangle(sample_image,
#                   tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
#                   tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
#                   (255, 0, 0), 2)
#
#     ax[idx].imshow(sample_image)

# Deep Learning
# chairtracker = deep_learning_fun(train, val)
chairtracker = load_model('chairtracker.h5')
# Live Tracking
live_detection(chairtracker)
