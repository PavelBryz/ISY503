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
from keras.src.layers import MaxPooling2D
from sklearn.model_selection import train_test_split

DATA_DIR = r"./assets/data"
TEST_SIZE = 0.2
DROPOUT_RATE = 0.5
NB_EPOCH = 10
SAMPLES_PER_EPOCH = 5000
BATCH_SIZE = 32
SAVE_BEST_ONLY = True
LEARNING_RATE = 1E-3
IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


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

    def _prepare_image(path: str):
        image = preprocess_img(load_image(Path(path)), image_width=image_width, image_height=image_height)
        if not is_eval:
            image = apply_augmentations(image)
        image = normalize_image(image)
        return image

    for epoch in count(0, 1):
        if is_eval and epoch >= 1:
            break

        images, angles = [], []
        for idx in np.random.permutation(x.shape[0]):
            center, left, right = x[['center', 'left', 'right']].iloc[idx]
            steering_angle = y['steering'].iloc[idx]

            images.append(_prepare_image(center))
            angles.append(steering_angle)

            if use_left:
                images.append(_prepare_image(left))
                angles.append(steering_angle)

            if use_right:
                images.append(_prepare_image(right))
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


def load_data() -> Dict[str, Any]:
    """
    Load training data and split it into training and validation set
    """
    data_dir = Path(DATA_DIR)
    driving_data = pd.read_csv(data_dir / 'driving_log.csv')
    X = driving_data[['center', 'left', 'right']]
    y = driving_data[['steering']]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=0)

    return dict(
        X_train=X_train, X_valid=X_valid,
        y_train=y_train, y_valid=y_valid
    )


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

def train_model(
        model: Sequential,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series
):
    """
    Train the model
    """
    file_path = get_file_path()
    checkpoint = ModelCheckpoint(file_path,
                                 monitor='val_loss',
                                 verbose=0,
                                 # save_best_only=True,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))
    model.fit_generator(batch_generator(X_train, y_train, batch_size=BATCH_SIZE,
                                        is_eval=False,
                                        use_left=False,
                                        use_right=False),
                        validation_data=batch_generator(X_valid, y_valid, batch_size=BATCH_SIZE,
                                                        is_eval=True,
                                                        use_left=False,
                                                        use_right=False),
                        steps_per_epoch=2000,
                        epochs=10,
                        callbacks=[checkpoint],
                        verbose=1)


def main():
    """
    Load train/validation data set and train the model
    """
    data = load_data()
    train_model(
        model=build_model(),
        **data
    )


if __name__ == '__main__':
    main()
