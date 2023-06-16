import tensorflow as tf
from init import load_image


def load_aug_data():

    train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
    train_images = train_images.map(load_image)
    train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
    train_images = train_images.map(lambda x: x / 255)
    print(len(train_images))

    test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
    test_images = test_images.map(load_image)
    test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
    test_images = test_images.map(lambda x: x / 255)
    print(len(test_images))

    val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
    val_images = val_images.map(load_image)
    val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
    val_images = val_images.map(lambda x: x / 255)
    print(len(val_images))

    return (train_images, test_images, val_images)

