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


def get_driving_data(data_dir):
    return pd.read_csv(data_dir / 'driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle',
                                                            'reverse', 'speed'])


def get_data_folder():
    data_dir = Path(__file__).parent / 'assets' / 'data'

    # Check if the file exists
    if data_dir.exists():
        print("File exists.")
    else:
        print("File doesn't exist or path is incorrect.")

    return data_dir


def get_checkpoint_filepath():
    result_folder = 'results'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    return os.path.join(result_folder, 'model-{epoch:03d}.h5')


def load_image(image_path: Path) -> np.ndarray:
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
    # Crop and resize the image
    image = image[60:-25, :, :]
    image = cv2.resize(image, (image_width, image_height), cv2.INTER_AREA)
    return image


def apply_augmentations(image: np.ndarray) -> np.ndarray:
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
    # Assert that the image is not empty
    assert image is not None, "Image is None, cannot normalize."
    return image / 255.0


def prepare_image(path: str, is_eval, image_width, image_height):
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
    driving_data = get_driving_data(data_dir)

    X = driving_data[['center', 'left', 'right']]
    y = driving_data[['steering']]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(activation_type, x_train):
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
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=LEARNING_RATE))


def create_train_generator(X_train, y_train):
    return batch_generator(X_train, y_train, batch_size=BATCH_SIZE,
                           is_eval=False, use_left=False, use_right=False)


def create_valid_generator(X_valid, y_valid):
    return batch_generator(X_valid, y_valid, batch_size=BATCH_SIZE,
                           is_eval=True, use_left=False, use_right=False)


def train_model(model, X_train, X_valid, y_train, y_valid):
    file_path = get_checkpoint_filepath()
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=0, mode='auto')

    compile_model(model)

    train_generator = create_train_generator(X_train, y_train)
    valid_generator = create_valid_generator(X_valid, y_valid)

    model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        steps_per_epoch=len(X_train) // BATCH_SIZE,
                        validation_steps=len(X_valid) // BATCH_SIZE,
                        epochs=10,
                        callbacks=[checkpoint],
                        verbose=1)


def main():
    data_dir = get_data_folder()

    X_train, X_valid, y_train, y_valid = load_data(data_dir)

    model = build_model('elu', X_train)

    train_model(
        model,
        X_train, X_valid, y_train, y_valid
    )


if __name__ == '__main__':
    main()
