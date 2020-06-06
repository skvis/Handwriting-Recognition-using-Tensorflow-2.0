import os

DATA_PATH = os.path.join(os.getcwd(), '../input/mnist.npz')   # '../input/mnist.npz'
MODEL_PATH = '../models/'

IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNEL = 1

TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
