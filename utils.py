from glob import glob
import os
from shutil import rmtree


def check_folder(folder, remove_previous=True):
    if os.path.exists(folder):
        if remove_previous:
            rmtree(folder)
            os.makedirs(folder)
    else:
        os.makedirs(folder)


def abs_path(path, *paths):
    return os.path.join(path, *paths)


def covid_path():
    return abs_path('dataset/covid')


def covid_masks_path():
    return abs_path('cov_masks')


def non_covid_path():
    return abs_path('dataset/normal')


def non_covid_masks_path():
    return abs_path('non_cov_masks')


def cov_processed_path():
    return abs_path('cov_processed')


def non_cov_processed_path():
    return abs_path('non_cov_processed')


def model_path():
    return abs_path('model.h5')


def covid_images():
    return glob(covid_path() + "/*g")


def non_covid_images():
    return glob(non_covid_path() + "/*g")


def covid_masks():
    return glob(covid_masks_path() + "/*g")


def non_covid_masks():
    return glob(non_covid_masks_path() + "/*g")


def cov_processed():
    return glob(cov_processed_path() + "/*g")


def non_cov_processed():
    return glob(non_cov_processed_path() + "/*g")
