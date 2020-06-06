import dataset
import config


def data_augmentation():
    train, test = dataset.load_dataset()

    train_images, train_labels = train
    test_images, test_labels = test

    train_images = train_images.reshape(60000,
                                        config.IMG_WIDTH,
                                        config.IMG_HEIGHT,
                                        config.IMG_CHANNEL)
    train_images = train_images / 255.0

    test_images = test_images.reshape(10000,
                                      config.IMG_WIDTH,
                                      config.IMG_HEIGHT,
                                      config.IMG_CHANNEL)
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels
