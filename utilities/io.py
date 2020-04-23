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
    """Read json file

    Args:
        json_file (str): Path of source json file

    Returns:
        dict: Dictionary representation of the contents of json_file
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    return data

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


