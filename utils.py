from glob import glob
import os
from shutil import rmtree
from vhviv_tools.json import json

# Project variables (just to avoid repeating/misspelling them)
# JOB VARIABLES:
WB_JOB_UPLOAD_DATASET = "upload_dataset"
WB_JOB_LOAD_DATASET = "load_dataset"
WB_JOB_LOG_TABLE = "log_interactive_table"
WB_JOB_HISTOGRAM_CHART = "log_histogram_chart"
WB_JOB_LOAD_ARTIFACTS = "load_artifacts"
WB_JOB_MODEL_FIT = "model_fit"
# TAG VARIABLES:
WB_ARTIFACT_DATASET_TAG = "dataset"
WB_ARTIFACT_COVID_TAG = "covid"
WB_ARTIFACT_COVID_MASKS_TAG = "covid_mask"
WB_ARTIFACT_COVID_PROCESSED_TAG = "covid_processed"
WB_ARTIFACT_NORMAL_TAG = "normal"
WB_ARTIFACT_NORMAL_MASKS_TAG = "normal_mask"
WB_ARTIFACT_NORMAL_PROCESSED_TAG = "normal_processed"
WB_ARTIFACT_MODEL_TAG = "model"


def load_config(key: str) -> str:
    """This function loads the configuration from the JSON file called and returns the value associated with the key argument."""
    return json("config.json")[key]


def __get_dataset_path(value: str) -> str:
    """The function reads the "config.json" file and returns the path for the specified dataset from the "datasets" key in the config file."""
    return load_config(WB_ARTIFACT_DATASET_TAG)[value]


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


def cov_masks_path():
    """This function returns the absolute path of the covid masks dataset."""
    return abs_path(__get_dataset_path("covid_masks_path"))


def normal_path():
    """This function returns the absolute path of the raw healthy dataset."""
    return abs_path(__get_dataset_path("raw_normal_path"))


def normal_masks_path():
    """This function returns the absolute path of the healthy masks dataset."""
    return abs_path(__get_dataset_path("normal_masks_path"))


def cov_processed_path():
    """This function returns the absolute path of the covid processed dataset."""
    return abs_path(__get_dataset_path("covid_processed_path"))


def normal_processed_path():
    """This function returns the absolute path of the healthy processed dataset."""
    return abs_path(__get_dataset_path("normal_processed_path"))


def model_path():
    """This function returns the absolute path of the model."""
    return abs_path(__get_dataset_path("model_path"))


def cov_images():
    """Returns a list of all files with the extension "g" from the directory specified by the function cov_path(). """
    return glob(cov_path() + "/*g")


def normal_images():
    """Returns a list of all files with the extension "g" from the directory specified by the function non_cov_path(). """
    return glob(normal_path() + "/*g")


def cov_masks():
    """Returns a list of all files with the extension "g" from the directory specified by the function covid_masks_path(). """
    return glob(cov_masks_path() + "/*g")


def normal_masks():
    """Returns a list of all files with the extension "g" from the directory specified by the function non_covid_masks_path()."""
    return glob(normal_masks_path() + "/*g")


def cov_processed():
    """Returns a list of all files with the extension "g" from the directory specified by the function covid_processed()."""
    return glob(cov_processed_path() + "/*g")


def normal_processed():
    """Returns a list of all files with the extension "g" from the directory specified by the function non-cov-processed-path()."""
    return glob(normal_processed_path() + "/*g")
