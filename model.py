from keras.layers import GlobalAveragePooling2D, InputLayer, Conv2D, MaxPooling2D, Dense, \
    Activation, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import os
import random
from itertools import count
from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 120


def preprocess():
    data_dir = Path(__file__).parent / 'assets' / 'data'
    print("test - ", data_dir)

    # Check if the file exists
    if data_dir.exists():
        print("File exists.")
    else:
        print("File doesn't exist or path is incorrect.")

    print(data_dir)
    driving_data = pd.read_csv(data_dir / 'updated_driving_log.csv')
    print(driving_data)
    x = driving_data[['center', 'left', 'right']]
    y = driving_data[['steering']]

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.30, random_state=0)

    return x_train, x_valid, y_train, y_valid


def build_model():
    model_built = Sequential()
    model_built.add(InputLayer(input_shape=(32, 32, 3)))

    # Define configurations for convolutional layers
    conv_configs = [
        {'filters': 8},
        {'filters': 16},
        {'filters': 32},
        {'filters': 64},
        {'filters': 128}
    ]

    for config in conv_configs:
        model_built.add(Conv2D(config['filters'], kernel_size=(3, 3), padding='same', use_bias=False))
        model_built.add(BatchNormalization())
        model_built.add(Activation('elu'))

        model_built.add(Conv2D(config['filters'], kernel_size=(3, 3), padding='same', use_bias=False))
        model_built.add(BatchNormalization())
        model_built.add(Activation('elu'))

        if config != conv_configs[-1]:  # Add MaxPooling except for the last configuration
            model_built.add(MaxPooling2D(pool_size=(2, 2)))

    model_built.add(GlobalAveragePooling2D())
    model_built.add(Dense(128, use_bias=False))
    model_built.add(BatchNormalization())
    model_built.add(Activation('elu'))
    model_built.summary()

    return model_built


def training(model, x_train, x_valid, y_train, y_valid):
    result_folder = 'results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    checkpoint = ModelCheckpoint(os.path.join(result_folder, 'model-{epoch:03d}.h5'),
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    optimizer = Adam(learning_rate=INIT_LR)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    train_generator = batch_generator(x_train, y_train, BATCH_SIZE,
                                      use_augmentations=True,
                                      use_left=False,
                                      use_right=False)

    valid_generator = batch_generator(x_valid, y_valid, BATCH_SIZE,
                                      use_augmentations=False,
                                      use_left=False,
                                      use_right=False)

    model.fit(train_generator,
              steps_per_epoch=len(x_train) // BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=valid_generator,
              validation_steps=len(x_valid) // BATCH_SIZE,
              callbacks=[checkpoint],
              verbose=1)


def batch_generator(
        x: pd.DataFrame,
        y: pd.Series,
        batch_size: int,
        use_augmentations: bool = False,
        use_left: bool = True,
        use_right: bool = True,
        image_height: int = 66,
        image_width: int = 200
):
    """
    Generator function that yields batches of images and corresponding steering angles.

    :param data: DataFrame containing image paths and steering angles.
    :param batch_size: Number of image sets to include in each batch.
    :param use_augmentations: Boolean indicating whether to apply augmentations.
    :param use_left: Boolean indicating whether to include left images.
    :param use_right: Boolean indicating whether to include right images.
    :yield: A batch of images and corresponding steering angles.
    """

    def _prepare_image(path: str):
        image = preprocess_img(load_image(Path(path)), image_width=image_width, image_height=image_height)
        if use_augmentations:
            image = apply_augmentations(image)
        image = normalize_image(image)
        return image

    for _ in count(0, 1):
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


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from a given path and convert it from BGR to RGB color space.

    :param image_path: The path to the image file.
    :return: The loaded image in RGB format or None if the path is invalid.
    """
    # Load image using OpenCV
    image = cv2.imread(str(image_path))

    # Convert BGR to RGB format
    if image is not None:
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


if __name__ == "__main__":
    # prepare the data for build the model
    X_train, X_valid, y_train, y_valid = preprocess()

    # Call build_model with the configurations
    model = build_model()

    # Training the data based on the model
    training(model, X_train, X_valid, y_train, y_valid)
