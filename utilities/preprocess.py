import cv2
import numpy as np

def pad_image(image: np.ndarray) -> np.ndarray:
    """Pad image to make it a square image

    Args:
        image (np.ndarray): Absolute path of image file

    Returns:
        np.ndarray: padded image
    """
    ht, wt = image.shape[:2]
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
    return padded_image

def read_and_pad(image_path: str) -> (np.ndarray, tuple, tuple):
    """Read and image and pad it to make it a square image

    Args:
        image_path (str): Absolute path of image file

    Returns:
        np.ndarray, tuple, tuple: padded image, original shape, new shape
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    original_shape = image.shape
    image = pad_image(image)
    new_shape = image.shape
    return image, original_shape, new_shape

# TODO: Should go into utilities repo
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

