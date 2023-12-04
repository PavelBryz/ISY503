from pathlib import Path

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, InputLayer, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from data_preprocessing import batch_generator

#np.random.seed(0)

INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 120

data_dir=Path('./beta_simulator_windows/data')
driving_data = pd.read_csv(data_dir / 'updated_driving_log.csv')

def load_data(data_dir=Path('./beta_simulator_windows/data')):
    driving_data = pd.read_csv(data_dir / 'updated_driving_log.csv')
    X = driving_data[['center', 'left', 'right']]
    y = driving_data[['steering']]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_valid, y_train, y_valid


def make_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(32, 32, 3)))

    model.add(Conv2D(8, kernel_size=(3, 3), input_shape=(32, 32, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(8, kernel_size=(3, 3), input_shape=(32, 32, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.summary()

    return model


def train_model(model, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=INIT_LR))

    # Prepare training and validation data using batch_generator
    train_generator = batch_generator(X_train, y_train, BATCH_SIZE,
                                      use_augmentations=True,
                                      use_left=False,
                                      use_right=False)

    valid_generator = batch_generator(X_valid, y_valid, BATCH_SIZE,
                                      use_augmentations=False,
                                      use_left=False,
                                      use_right=False)

    model.fit_generator(train_generator,
                        steps_per_epoch=len(X_train) // BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=valid_generator,
                        validation_steps=len(X_valid) // BATCH_SIZE,
                        callbacks=[checkpoint],
                        verbose=1)


def main():
    data = load_data(data_dir)
    model = make_model()
    train_model(model, *data)


if __name__ == '__main__':
    main()
