import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import os
from pathlib import Path

np.random.seed(0)

TEST_SIZE = 0.2
KEEP_PROB = 0.5
NB_EPOCH = 10
SAMPLES_PER_EPOCH = 20000
BATCH_SIZE = 40
SAVE_BEST_ONLY = 'true'
LEARNING_RATE = 1.0e-4

def get_data_folder():
    data_dir = Path(__file__).parent / 'assets' / 'data'

    # Check if the file exists
    if data_dir.exists():
        print("File exists.")
    else:
        print("File doesn't exist or path is incorrect.")

    return data_dir


def get_driving_data(data_dir):
    return pd.read_csv(data_dir / 'driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle',
                                                            'reverse', 'speed'])

def get_file_path():
    result_folder = 'results'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    return os.path.join(result_folder, 'model-{epoch:03d}.h5')

def load_data(data_dir):
    driving_data = get_driving_data(data_dir)

    X = driving_data[['center', 'left', 'right']].values
    y = driving_data['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(activation_type, kernel_size, pool_size, strides):
    model = Sequential([
        Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE),
        Conv2D(32, kernel_size, activation=activation_type, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size, strides),
        Conv2D(64, kernel_size, activation=activation_type),
        MaxPooling2D(pool_size, strides),
        Conv2D(128, kernel_size, activation=activation_type),
        MaxPooling2D(pool_size, strides),
        Flatten(),
        Dense(128, activation=activation_type),
        Dropout(0.5),
        Dense(1)
    ])
    model.summary()

    return model


def train_model(model, data_dir, X_train, y_train):
    checkpoint = ModelCheckpoint(get_file_path(),
                                 monitor='val_loss',
                                 verbose=0,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))

    model.fit_generator(batch_generator(data_dir, X_train, y_train, BATCH_SIZE, True),
                        SAMPLES_PER_EPOCH,
                        NB_EPOCH,
                        callbacks=[checkpoint],
                        verbose=1)

def main():
    data_dir = get_data_folder()

    #load data
    X_train, X_valid, y_train, y_valid = load_data(data_dir)

    #create the model
    model = build_model('relu', (3, 3), 2, 2)

    #train model on data, it saves as model.h5
    train_model(model, data_dir, X_train, y_train)


if __name__ == '__main__':
    main()

