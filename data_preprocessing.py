import random
from itertools import count
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd


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

    def _prepare_image(image, ):
        image = preprocess_img(load_image(image), image_width=image_width, image_height=image_height)
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


if __name__ == "__main__":
    from data_processing_testing import load_data

    X_train, X_valid, y_train, y_valid = load_data()
    for images, angles in batch_generator(
            X_train, y_train,
            batch_size=32,
            use_augmentations=False,
            use_left=True,
            use_right=True,
            image_height=200,
            image_width=350
    ):
        images = (images * 255).astype(np.uint8)
        for angle, image in zip(angles, images):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.putText(
                image,
                f"{float(angle):.2f}",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA
            )

            cv2.imshow("image", image)
            cv2.waitKey(0)
