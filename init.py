# --------------------------------------------------------------------------
# Init functions
import tensorflow as tf
from matplotlib import pyplot as plt


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


def init():
    gpus = tf.config.experimental.list_physical_devices('CPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.list_physical_devices('CPU')
    images = tf.data.Dataset.list_files('photos\\*.jpg', shuffle=False)
    images = images.map(load_image)
    images.as_numpy_iterator().next()
    image_generator = images.batch(4).as_numpy_iterator()
    plot_images = image_generator.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, image in enumerate(plot_images):
        ax[idx].imshow(image)
    plt.show()
