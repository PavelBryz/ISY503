import os
import random
from itertools import count
from pathlib import Path
from typing import Any, Dict
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# constants
TEST_SIZE = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 1E-3


def get_driving_data(data_dir):
    '''
    Get the data from csv file
    :param data_dir: data directory
    :return: data
    '''
    return pd.read_csv(data_dir / 'driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle',
                                                            'reverse', 'speed'])


def get_data_folder():
    '''
    get the data folder
    :return: data folder path
    '''
    data_dir = Path(__file__).parent / 'assets' / 'data'

    # Check if the file exists
    if data_dir.exists():
        print("File exists.")
    else:
        print("File doesn't exist or path is incorrect.")

    return data_dir


def get_checkpoint_filepath():
    '''
    create the file model.h5
    verify the results folder exist if not create it
    :return: return the model.h5 filepath
    '''
    result_folder = 'results'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    return os.path.join(result_folder, 'model-{epoch:03d}.h5')


def load_image(image_path: Path) -> np.ndarray:
    """
        Load an image from a given path and convert it from BGR to RGB color space.

        :param image_path: The path to the image file.
        :return: The loaded image in RGB format or None if the path is invalid.
    """

    # Load image using OpenCV
    image = cv2.imread(str(image_path))
    assert image is not None, f"{image_path}"

    # Convert BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def preprocess_img(
        image: np.ndarray,
        image_width: int,
        image_height: int
) -> np.ndarray:
    """
        Preprocess the image by cropping and resizing it.

        :param image: The input image.
        :return: The preprocessed image.
    """

    # Crop and resize the image
    image = image[60:-25, :, :]
    image = cv2.resize(image, (image_width, image_height), cv2.INTER_AREA)

    return image


def apply_augmentations(image: np.ndarray) -> np.ndarray:
    """
        Apply a series of augmentations to the image to enhance the dataset variety.

        :param image: The input image.
        :return: The augmented image.
    """

    # Seed for reproducibility of augmentations
    random.seed(42)

    # Define a series of augmentations
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0,
                           border_mode=cv2.BORDER_CONSTANT, value=(127, 127, 127)),
        A.CLAHE(clip_limit=2),
        A.Blur(blur_limit=3),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.MotionBlur(p=.2),
        A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=(127, 127, 127)),
        A.GridDistortion(p=.1, border_mode=cv2.BORDER_CONSTANT, value=(127, 127, 127)),
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(),
    ])

    # Apply the transformations
    return transform(image=image)['image']


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
            Normalize the image by scaling pixel values to the range [0, 1].

            :param image: The input image.
            :return: Normalized image.
    """

    # Assert that the image is not empty
    assert image is not None, "Image is None, cannot normalize."
    return image / 255.0


def prepare_image(path: str, is_eval, image_width, image_height):
    '''
    Prepares an image for processing based on the given parameters.

    :param path: The path to the image file.
    :param is_eval: A boolean flag indicating whether the image is for evaluation (True) or training (False).
    :param image_width: The desired width of the image.
    :param image_height: The desired height of the image.
    :return: The prepared image ready for further processing.
    '''
    image = preprocess_img(load_image(Path(path)), image_width=image_width, image_height=image_height)
    if not is_eval:
        image = apply_augmentations(image)
    image = normalize_image(image)
    return image


def batch_generator(
        x: pd.DataFrame,
        y: pd.Series,
        batch_size: int,
        is_eval: bool = False,
        use_left: bool = True,
        use_right: bool = True,
        image_height: int = 66,
        image_width: int = 200
):
    """
        Generator function that yields batches of images and corresponding steering angles.

        :param data: DataFrame containing image paths and steering angles.
        :param batch_size: Number of image sets to include in each batch.
        :param is_eval: ...
        :param use_left: Boolean indicating whether to include left images.
        :param use_right: Boolean indicating whether to include right images.
        :yield: A batch of images and corresponding steering angles.
    """

    for epoch in count(0, 1):
        if is_eval and epoch >= 1:
            break

        images, angles = [], []
        for idx in np.random.permutation(x.shape[0]):
            center, left, right = x[['center', 'left', 'right']].iloc[idx]
            steering_angle = y['steering'].iloc[idx]

            images.append(prepare_image(center, is_eval, image_width, image_height))
            angles.append(steering_angle)

            if use_left:
                images.append(prepare_image(left, is_eval, image_width, image_height))
                angles.append(steering_angle)

            if use_right:
                images.append(prepare_image(right, is_eval, image_width, image_height))
                angles.append(steering_angle)

            if len(images) >= batch_size:
                images_np = np.stack(images[:batch_size], axis=0)
                angles_np = np.array(angles[:batch_size], dtype=np.float32)

                indices = np.arange(images_np.shape[0])
                np.random.shuffle(indices)
                images_np = images_np[indices]
                angles_np = angles_np[indices]

                yield images_np, angles_np

                images.clear()
                angles.clear()


def load_data(data_dir) -> Dict[str, Any]:
    '''
    Load training data and split it into training and validation set
    :param data_dir:  data directory
    :return: the x and y train and valid
    '''
    driving_data = get_driving_data(data_dir)

    X = driving_data[['center', 'left', 'right']]
    y = driving_data[['steering']]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(activation_type):
    '''
    Creates a convolutional neural network model based on NVIDIA's architecture.

    :param activation_type: The type of activation function to be used in the model's layers.
    :return: A convolutional neural network model based on the specified activation function.
    '''

    model = Sequential([
        Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)),
        Conv2D(24, (5, 5), strides=(2, 2), activation=activation_type, input_shape=(28, 28, 1)),
        Conv2D(36, (5, 5), strides=(2, 2), activation=activation_type),
        Conv2D(48, (5, 5), strides=(2, 2), activation=activation_type),
        Conv2D(64, (3, 3), padding="same", activation=activation_type),
        Conv2D(64, (3, 3), padding="same", activation=activation_type),
        Dropout(0.3),
        Flatten(),
        Dense(100, activation=activation_type),
        Dense(50, activation=activation_type),
        Dense(10, activation=activation_type),
        Dense(1)
    ])

    model.summary()

    return model


def compile_model(model):
    '''
    Compiles the given neural network model using specified loss function and optimizer.

    :param model: The neural network model to be compiled.
    '''
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=LEARNING_RATE))

def create_data_generator(X_data, y_data, is_eval=False):
    '''
    Creates a data generator using batch_generator for training or validation.

    :param X_data: The dataset (input features).
    :param y_data: The dataset (target labels).
    :param is_eval: A flag indicating whether the generator is for evaluation (default: False).
    :return: A generator for data batches.
    '''
    return batch_generator(X_data, y_data, batch_size=BATCH_SIZE,
                           is_eval=is_eval, use_left=False, use_right=False)


def train_model(model, X_train, X_valid, y_train, y_valid):
    '''
       Trains the given model using the provided training and validation datasets.

       :param model: The neural network model to be trained.
       :param X_train: The training dataset (input features).
       :param X_valid: The validation dataset (input features).
       :param y_train: The training dataset (target labels).
       :param y_valid: The validation dataset (target labels).
       :return: None
    '''

    file_path = get_checkpoint_filepath()
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=0, mode='auto')

    compile_model(model)

    train_generator = create_data_generator(X_train, y_train)
    valid_generator = create_data_generator(X_valid, y_valid, is_eval=True)

    model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        steps_per_epoch=len(X_train) // BATCH_SIZE,
                        validation_steps=len(X_valid) // BATCH_SIZE,
                        epochs=10,
                        callbacks=[checkpoint],
                        verbose=1)


def main():
    '''
        Main function to execute the training process of a neural network model.

        This function performs the following steps:
        1. Retrieves the data directory using get_data_folder().
        2. Loads the training and validation datasets using load_data().
        3. Builds a neural network model using build_model() with 'elu' as the activation function.
        4. Trains the model using train_model() with the loaded datasets.

    '''

    data_dir = get_data_folder()

    X_train, X_valid, y_train, y_valid = load_data(data_dir)

    model = build_model('elu')

    train_model(
        model,
        X_train, X_valid, y_train, y_valid
    )


if __name__ == '__main__':
    main()
