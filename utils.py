from glob import glob
import os
from shutil import rmtree
from vhviv_tools.json import json


def __get_dataset_path(value: str) -> str:
    """The function reads the "config.json" file and returns the path for the specified dataset from the "datasets" key in the config file."""
    config = json("config.json")
    return config["datasets"][value]


def check_folder(folder, remove_previous=True):
    """This function checks if a folder exists and creates it if it does not. If the folder already exists, then the function will remove the previous folder and create a new one if the remove_previous argument is set to True (which is its default value).
    
    Parameters: 
    folder (str): The path of the folder to check/create. 
    remove_previous (bool): A boolean value indicating whether to remove the previous folder if it already exists. Defaults to True. """
    if os.path.exists(folder):
        if remove_previous:
            rmtree(folder)
            os.makedirs(folder)
    else:
        os.makedirs(folder)


def abs_path(path, *paths):
    """This function takes in two arguments, path and *paths. It returns the absolute path of the given paths by joining them using the os.path.join() method. The *paths argument is a tuple of paths that will be joined to the path argument."""
    return os.path.join(path, *paths)


def dataset_path():
    """This function returns the absolute path of the raw dataset."""
    return abs_path(__get_dataset_path("raw_datasets_path"))


def cov_path():
    """This function returns the absolute path of the raw covid dataset."""
    return abs_path(__get_dataset_path("raw_covid_dataset_path"))


def covid_masks_path():
    """This function returns the absolute path of the covid masks dataset."""
    return abs_path(__get_dataset_path("covid_masks_path"))


def non_cov_path():
    """This function returns the absolute path of the raw healthy dataset."""
    return abs_path(__get_dataset_path("raw_normal_path"))


def non_covid_masks_path():
    """This function returns the absolute path of the healthy masks dataset."""
    return abs_path(__get_dataset_path("normal_masks_path"))


def cov_processed_path():
    """This function returns the absolute path of the covid processed dataset."""
    return abs_path(__get_dataset_path("covid_processed_path"))


def non_cov_processed_path():
    """This function returns the absolute path of the healthy processed dataset."""
    return abs_path(__get_dataset_path("normal_processed_path"))


def model_path():
    """This function returns the absolute path of the model."""
    return abs_path(__get_dataset_path("model_path"))


def cov_images():
    """Returns a list of all files with the extension "g" from the directory specified by the function cov_path(). """
    return glob(cov_path() + "/*g")


def non_cov_images():
    """Returns a list of all files with the extension "g" from the directory specified by the function non_cov_path(). """
    return glob(non_cov_path() + "/*g")


def cov_masks():
    """Returns a list of all files with the extension "g" from the directory specified by the function covid_masks_path(). """
    return glob(covid_masks_path() + "/*g")


def non_cov_masks():
    """Returns a list of all files with the extension "g" from the directory specified by the function non_covid_masks_path()."""
    return glob(non_covid_masks_path() + "/*g")


def cov_processed():
    """Returns a list of all files with the extension "g" from the directory specified by the function covid_processed()."""
    return glob(cov_processed_path() + "/*g")


def non_cov_processed():
    """Returns a list of all files with the extension "g" from the directory specified by the function non-cov-processed-path()."""
    return glob(non_cov_processed_path() + "/*g")
