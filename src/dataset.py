import tensorflow as tf
import config


def load_dataset():
    mnist = tf.keras.datasets.mnist
    train, test = mnist.load_data(path=config.DATA_PATH)

    return train, test
