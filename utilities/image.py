import cv2
import numpy as np


def pad_image(image: np.ndarray) -> np.ndarray:
    """Pad image to make it a square image

    Args:
        image (np.ndarray): numpy array

    Returns:
        np.ndarray: padded image
    """
    ht, wt = image.shape[:2]
    original_shape = (ht, wt)
    final_shape = max(ht,wt)
    small_shape = min(ht,wt)

    padding = np.array([(0, 0), (0, 0)])
    min_axis = np.argmin(image.shape[:2])

    pad_value = (final_shape - small_shape) // 2
    padding[min_axis] = np.array((pad_value, pad_value))

    if len(image.shape) != 2:
        padding = padding.tolist()
        padding.append((0, 0))
        padding = np.array(padding)

    padded_image = np.pad(image, padding, mode='constant')
    return padded_image, original_shape, padded_image.shape[:2]

def pad_points(points_2d: np.ndarray, image_shape: tuple) -> np.ndarray:
    """Pad keypoints of the image

    Args:
        points_2d (np.ndarray): 2D keypoints
        image_shape (tuple): (h, w) shape of the image

    Returns:
        np.ndarray: Padded keypoints
    """
    ht, wt = image_shape[:2]
    smaller_shape = min(ht,wt)
    final_shape = max(ht,wt)
    pad_length = int((final_shape - smaller_shape)/2)

    if ht == final_shape:
        axis = 0
    elif wt == final_shape:
        axis = 1

    if len(points_2d.shape) == 3:
        points_2d[:, :, axis] += pad_length
    else:
        points_2d[:, axis] += pad_length

    return points_2d

def normalize(image: np.ndarray, range='0to1'):
    if range == '0to1':
        image = image.astype(np.float) / 255.
    elif range == '-1to1':
        image = (image.astype(np.float) / 255) * 2.0 - 1.0
    else:
        raise NotImplementedError

    return image
