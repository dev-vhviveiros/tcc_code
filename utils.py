import os
from shutil import rmtree
from vhviv_tools import json

CONFIG_JSON = None

def load_config(key: str) -> str:
    """
    Load a value from the configuration file.

    Args:
        key: The key to look up in the configuration file.

    Returns:
        The value associated with the key in the configuration file.
    """
    global CONFIG_JSON
    if not CONFIG_JSON:
        CONFIG_JSON = json.load("config.json")
    return CONFIG_JSON[key]


def check_folder(folder: str, clear_folder: bool = True):
    """
    Check if a folder exists and create it if it does not.

    If the folder already exists, the function will remove the previous folder and create a new one if the `clear_folder`
    argument is set to True (which is its default value).

    Args:
        folder: The path of the folder to check/create.
        clear_folder: A boolean value indicating whether to remove the previous folder if it already exists.
            Defaults to True.
    """
    if os.path.exists(folder):
        if clear_folder:
            rmtree(folder)
            os.makedirs(folder)
    else:
        os.makedirs(folder)


def abs_path(path: str, *subpaths: str) -> str:
    """
    Get the absolute path of a file or directory.

    Args:
        path: The path to the file or directory.
        subpaths: Additional path segments to join to the path.

    Returns:
        The absolute path of the file or directory.
    """
    return os.path.join(path, *subpaths)