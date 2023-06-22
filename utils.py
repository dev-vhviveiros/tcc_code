from glob import glob
import os
from shutil import rmtree
from vhviv_tools.json import json

# TAG VARIABLES:
DATASET_TAG = "dataset"
COVID_TAG = "covid"
MODEL_TAG = "model"
CHARACTERISTICS_TAG = "characteristics"
COVID_MASKS_TAG = "covid_mask"
COVID_PROCESSED_TAG = "covid_processed"
NORMAL_TAG = "normal"
NORMAL_MASKS_TAG = "normal_mask"
NORMAL_PROCESSED_TAG = "normal_processed"


def load_config(key: str) -> str:
    """
    Load a value from the configuration file.

    Args:
        key: The key to look up in the configuration file.

    Returns:
        The value associated with the key in the configuration file.
    """
    return json("config.json")[key]


def load_wdb_config(key: str) -> str:
    """
    Load a value from the configuration file, with the "wandb" key.

    Args:
        key: The key to look up in the W&B configuration.

    Returns:
        The value associated with the key in the W&B configuration.
    """
    return load_config("wandb")[key]


def get_dataset_path(value: str) -> str:
    """
    Get the path to a dataset.

    Args:
        value: The name of the dataset to get the path for.

    Returns:
        The absolute path to the specified dataset.
    """
    return load_config(DATASET_TAG)[value]


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


def dataset_path() -> str:
    """
    Get the absolute path of the raw dataset.

    Returns:
        The absolute path of the raw dataset.
    """
    return abs_path(get_dataset_path("raw_datasets_path"))


def cov_path() -> str:
    """
    Get the absolute path of the raw covid dataset.

    Returns:
        The absolute path of the raw covid dataset.
    """
    return abs_path(get_dataset_path("raw_covid_dataset_path"))


def cov_masks_path() -> str:
    """
    Get the absolute path of the covid masks dataset.

    Returns:
        The absolute path of the covid masks dataset.
    """
    return abs_path(get_dataset_path("covid_masks_path"))


def normal_path() -> str:
    """
    Get the absolute path of the raw healthy dataset.

    Returns:
        The absolute path of the raw healthy dataset.
    """
    return abs_path(get_dataset_path("raw_normal_path"))


def normal_masks_path() -> str:
    """
    Get the absolute path of the healthy masks dataset.

    Returns:
        The absolute path of the healthy masks dataset.
    """
    return abs_path(get_dataset_path("normal_masks_path"))


def cov_processed_path() -> str:
    """
    Get the absolute path of the covid processed dataset.

    Returns:
        The absolute path of the covid processed dataset.
    """
    return abs_path(get_dataset_path("covid_processed_path"))


def normal_processed_path() -> str:
    """
    Get the absolute path of the healthy processed dataset.

    Returns:
        The absolute path of the healthy processed dataset.
    """
    return abs_path(get_dataset_path("normal_processed_path"))


def model_path() -> str:
    """
    Get the absolute path of the model.

    Returns:
        The absolute path of the model.
    """
    return load_config("others")["model_path"]


def cov_images(extension: str = "g") -> list[str]:
    """
    Get a list of all files with the specified extension from the covid dataset directory.

    Args:
        extension: The file extension to search for. Defaults to "g".

    Returns:
        A list of all files with the specified extension from the covid dataset directory.
    """
    return glob(cov_path() + f"/*{extension}")


def normal_images(extension: str = "g") -> list[str]:
    """
    Get a list of all files with the specified extension from the healthy dataset directory.

    Args:
        extension: The file extension to search for. Defaults to "g".

    Returns:
        A list of all files with the specified extension from the healthy dataset directory.
    """
    return glob(normal_path() + f"/*{extension}")


def cov_masks(extension: str = "g") -> list[str]:
    """
    Get a list of all files with the specified extension from the covid masks directory.

    Args:
        extension: The file extension to search for. Defaults to "g".

    Returns:
        A list of all files with the specified extension from the covid masks directory.
    """
    return glob(cov_masks_path() + f"/*{extension}")


def normal_masks(extension: str = "g") -> list[str]:
    """
    Get a list of all files with the specified extension from the healthy masks directory.

    Args:
        extension: The file extension to search for. Defaults to "g".

    Returns:
        A list of all files with the specified extension from the healthy masks directory.
    """
    return glob(normal_masks_path() + f"/*{extension}")


def cov_processed(extension: str = "g") -> list[str]:
    """
    Get a list of all files with the specified extension from the covid processed directory.

    Args:
        extension: The file extension to search for. Defaults to "g".

    Returns:
        A list of all files with the specified extension from the covid processed directory.
    """
    return glob(cov_processed_path() + f"/*{extension}")


def normal_processed(extension: str = "g") -> list[str]:
    """
    Get a list of all files with the specified extension from the healthy processed directory.

    Args:
        extension: The file extension to search for. Defaults to "g".

    Returns:
        A list of all files with the specified extension from the healthy processed directory.
    """
    return glob(normal_processed_path() + f"/*{extension}")


def characteristics_path() -> str:
    """
    Get the path to the characteristics file.

    Returns:
        The path to the characteristics file.
    """
    return load_config("other")["characteristics_path"]
