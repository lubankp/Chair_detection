import json
import tensorflow as tf

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
    return [label['class']], label['bbox']


def train_labels_fun():
    train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
    print(len(train_labels))

    test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
    print(len(test_labels))

    val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
    val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
    print(len(val_labels))

    return (train_labels, test_labels, val_labels)