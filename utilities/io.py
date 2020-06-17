"""Contains all standard function wrappers related to file reading and writing
"""
import gzip
import json
import numpy as np


def load_npz(dest_path: str) -> np.ndarray:
    """
    Args:
        dest_path (str): Path of npz file

    Returns:
        np.ndarray:
    """
    if '.gz' in dest_path:
        f = gzip.GzipFile(dest_path, 'r')
    else:
        f = gzip.GzipFile(dest_path+'.gz', "r")
    np_arr = np.load(f)
    return np_arr


def read_json(json_file: str) -> dict:
    """Reads json file

    Args:
        json_file (str): Path of source json file

    Returns:
        dict: Dictionary representation of the contents of json_file
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    return data


def load_pkl(src_path, t='rb', encoding='ASCII'):
    with open(src_path, t) as f:
        pkl = pickle.load(f, encoding=encoding)

    return pkl


def write_pkl(pkl, dest_path, t='wb', protocol=None):
    with open(dest_path, 'wb') as f:
        pickle.dump(pkl, f, protocol=protocol)


def write_json(dictionary: dict, file_path: str) -> None:
    """Write dictionary to json file

    Args:
        dictionary (dict): Dictionary
        file_path (str): Path of destination file
    """
    with open(file_path, 'w') as fp:
        json.dump(dictionary, fp)


def read_lines(file_path: str) -> list:
    """Given file, read each line and return a list of lines

    Args:
        file_path (str): Path of source file

    Returns:
        list: List of line contents of the file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [line.rstrip() for line in lines]

    return lines


def write_lines(lines: list, file_path: str) -> None:
    """Write list to file, line by line

    Args:
        lines (list): List that needs to be written to the file
        file_path (str): Path of destination file
    """
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(str(line) + '\n')


