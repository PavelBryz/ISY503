import pandas as pd  # data analysis toolkit - create, read, update, delete datasets
import numpy as np  # matrix math
from sklearn.model_selection import train_test_split  # to split out training and testing data
# keras is a high level wrapper on top of tensorflow (machine learning library)
# The Sequential container is a linear stack of layers
from keras.models import Sequential
# popular optimization strategy that uses gradient descent
from keras.optimizers import Adam
# to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
# what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
# helper class to define input shape and generate training images given image paths & steering angles
from utils import INPUT_SHAPE, batch_generator
# for command line arguments
import argparse
# for reading files
import os
from pathlib import Path

# for debugging, allows for reproducible (deterministic) results
np.random.seed(0)


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


def load_data():
    """
    Load training data and split it into training and validation set
    """

    # get the data folder
    data_dir = get_data_folder()

    # get the csv file
    driving_data = get_driving_data(data_dir)

    # yay dataframes, we can select rows and columns by their names
    # we'll store the camera images as our input data
    X = driving_data[['center', 'left', 'right']].values
    # and our steering commands as our output data
    y = driving_data['steering'].values

    # now we can split the data into a training (80), testing(20), and validation set
    # thanks scikit learn
    x_train, x_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.30,
                                                          random_state=0)

    return x_train, x_valid, y_train, y_valid


def build_model():
    '''
        creates a model based on sequential model
        return: return the model created
    '''

    activation_type = 'relu'
    kernel_size = (3, 3)
    pool_size = 2
    strides = 2

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

def get_file_path():
    result_folder = 'results'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    return os.path.join(result_folder, 'model-{epoch:03d}.h5')

def train_model(model, x_train, x_valid, y_train, y_valid):
    """
    Train the model
    """
    learning_rate = 1.0e-4

    file_path = get_file_path()
    # Saves the model after every epoch.
    # quantity to monitor, verbosity i.e logging mode (0 or 1),
    # if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    # mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc,
    # this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint(file_path,
                                 monitor='val_loss',
                                 verbose=0,
                                 mode='auto')

    # calculate the difference between expected steering angle and actual steering angle
    # square the difference
    # add up all those differences for as many data points as we have
    # divide by the number of them
    # that value is our mean squared error! this is what we want to minimize via
    # gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

    # Fits the model on data generated batch-by-batch by a Python generator.

    # The generator is run in parallel to the model, for efficiency.
    # For instance, this allows you to do real-time data augmentation on images on CPU in
    # parallel to training your model on GPU.
    # so we reshape our data into their appropriate batches and train our model simulatenously
    data_dir = get_data_folder()
    batch_size = 40
    samples_per_epoch = 20000
    nb_epoch = 10

    model.fit_generator(batch_generator(data_dir, x_train, y_train, batch_size, True),
                        samples_per_epoch,
                        nb_epoch,
                        # validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
                        callbacks=[checkpoint],
                        verbose=1)


def main():
    # load data
    data = load_data()
    # build model
    model = build_model()
    # train model on data, it saves as model.h5
    train_model(model, *data)


if __name__ == '__main__':
    main()
