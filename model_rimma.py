import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from data_preprocessing import batch_generator

# for debugging, allows for reproducible (deterministic) results
np.random.seed(0)


def load_data(
        config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Load training data and split it into training and validation set
    """
    # reads CSV file into a single dataframe variable
    data_dir = Path(config["data_directory"])
    driving_data = pd.read_csv(data_dir / 'updated_driving_log.csv')

    # yay dataframes, we can select rows and columns by their names
    # we'll store the camera images as our input data
    X = driving_data[['center', 'left', 'right']]
    # and our steering commands as our output data
    y = driving_data[['steering']]

    # now we can split the data into a training (80), testing(20), and validation set
    # thanks scikit learn
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=0)

    return dict(
        X_train=X_train, X_valid=X_valid,
        y_train=y_train, y_valid=y_valid
    )


def build_model(
        config: Dict[str, Any]
) -> Sequential:
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    input_shape = config["image_height"], config["image_width"], 3
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))
    model.add(Conv2D(24, 5, 5, activation='elu'))
    # model.add(Conv2D(36, 5, 5, activation='elu'))
    # model.add(Conv2D(48, 5, 5, activation='elu'))
    # model.add(Conv2D(64, 3, 3, activation='elu'))
    # model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(config["dropout_rate"]))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(
        model: Sequential,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series
):
    """
    Train the model
    """
    # Saves the model after every epoch.
    # quantity to monitor, verbosity i.e logging mode (0 or 1),
    # if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    # mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc,
    # this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    # calculate the difference between expected steering angle and actual steering angle
    # square the difference
    # add up all those differences for as many data points as we have
    # divide by the number of them
    # that value is our mean squared error! this is what we want to minimize via
    # gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=config["learning_rate"]))

    # Fits the model on data generated batch-by-batch by a Python generator.

    # The generator is run in parallel to the model, for efficiency.
    # For instance, this allows you to do real-time data augmentation on images on CPU in
    # parallel to training your model on GPU.
    # so we reshape our data into their appropriate batches and train our model simulatenously
    model.fit_generator(batch_generator(X_train, y_train, batch_size=config["batch_size"],
                                        is_eval=False,
                                        use_left=False,
                                        use_right=False),
                        validation_data=batch_generator(X_valid, y_valid, batch_size=config["batch_size"],
                                                        is_eval=True,
                                                        use_left=False,
                                                        use_right=False),
                        steps_per_epoch=5000 // 32,
                        epochs=50,
                        callbacks=[checkpoint],
                        verbose=1)


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('config_dir', type=str, help='path to config file')
    config = json.load(open(parser.parse_args().config_dir, "r"))

    data = load_data(config)
    train_model(
        model=build_model(config),
        config=config,
        **data
    )


if __name__ == '__main__':
    main()
