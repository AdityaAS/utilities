import numpy as np
import torch


def fig_to_numpy(fig) -> np.ndarray:
    """Converts matplotlib figure to numpy array

    Args:
        fig (TYPE): matplotlib.pyplot.figure object

    Returns:
        np.ndarray
    """
    fig.canvas.draw()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def make_one_hot(position: int, length: int) -> torch.Tensor:
    """
    Makes a 1D one hot vector of length l with a one at index position, rest all zeroes

    Arguments:
        position (int): index at which the one hot vector would have one
        l (int): length of the one hot vector

    Returns:
        torch.Tensor

    Usage:
        make_one_hot(position=3, l=5) -> torch.Tensor([0, 0, 0, 1, 0])
    """
    one_hot_vec = torch.zeros(length)
    one_hot_vec[position] = 1

    return one_hot_vec
