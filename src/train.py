from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import numpy as np
import model
import config
import data_preprocessing


def train_fn():

    cnn_model = model.CNN()
    # print(model.summary())

    cnn_model.compile(optimizer=RMSprop(lr=config.LEARNING_RATE),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    train_images, train_labels, test_images, test_labels = data_preprocessing.data_augmentation()

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    csv_logger = CSVLogger(f'{config.MODEL_PATH}training.log', separator=',', append=False)

    callbacks = [earlystop, learning_rate_reduction, csv_logger]

    history = cnn_model.fit(train_images, train_labels,
                            epochs=config.NUM_EPOCHS,
                            verbose=2,
                            callbacks=callbacks)

    print(cnn_model.evaluate(test_images, test_labels))

    cnn_model.save(f"{config.MODEL_PATH}my_model.h5")
    np.save(f'{config.MODEL_PATH}my_history.npy', history.history)


if __name__ == '__main__':
    train_fn()
