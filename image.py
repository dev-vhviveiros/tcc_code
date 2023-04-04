import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from glob import glob
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import mahotas.features.texture as mt
from typing import List
from numba import njit, prange

from sklearn.model_selection import train_test_split
import tqdm
from models import unet_model

from utils import abs_path


class Image:
    def __init__(self, file_path, divide=False, reshape=False):
        """
        Load and preprocess image file.

        Args:
            file_path (str): path to image file
            divide (bool): whether to divide the image by 255 after loading
            reshape (bool): whether to reshape the image to 1D array

        Attributes:
            path (str): path to image file
            divide (bool): whether to divide the image by 255 after loading
            reshape (bool): whether to reshape the image to 1D array
            data (ndarray): preprocessed image data
        """
        self.path = file_path
        self.divide = divide
        self.reshape = reshape
        self.data = self.__load_file()

    def __load_file(self, target_size=(512, 512)):
        """
        Load image file, preprocess and return the data.

        Args:
            target_size (tuple): target size to resize the image

        Returns:
            ndarray: preprocessed image data
        """
        # load image in grayscale
        img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

        # divide image by 255
        if self.divide:
            img = img / 255

        # resize image
        img = cv2.resize(img, target_size)

        # reshape image
        if self.reshape:
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)

        return img

    def get_file_dir(self):
        """
        Return the directory of the file.

        Returns:
            tuple: directory of the file
        """
        return os.path.splitext(os.path.basename(self.path))

    def save_to(self, path_dir):
        """
        Save the image to a file.

        Args:
            path_dir (str): directory to save the file
        """
        filename, fileext = self.get_file_dir()
        result_file = abs_path(path_dir, "%s_processed%s" % (filename, fileext))
        cv2.imwrite(result_file, self.data)

    def shape(self):
        """
        Return the shape of the image data.

        Returns:
            tuple: shape of the image data
        """
        return self.data.shape

    def hist(self):
        """
        Calculate and return the histogram of the image data.

        Returns:
            ndarray: histogram of the image data
        """
        # calculate histogram
        result = np.squeeze(cv2.calcHist([self.data], [0], None, [255], [1, 256]))

        # convert result to integer type
        result = np.asarray(result, dtype='int32')

        return result

    def save_hist(self, save_folder=''):
        """
        Save the histogram of the image data to a file.

        Args:
            save_folder (str): directory to save the histogram file
        """
        # plot histogram
        plt.figure()
        histg = cv2.calcHist([self.data], [0], None, [254], [1, 255])
        plt.plot(histg)

        # save histogram to file
        filename, _ = self.get_file_dir()
        result_file = abs_path(save_folder, "%s_histogram%s" % (filename, '.png'))
        plt.savefig(result_file)
        plt.close()

    def haralick(self):
        """
        Calculate and return the mean of Haralick texture features for 4 types of adjacency.

        Returns:
            ndarray: Mean of the Haralick texture features.
        """
        # Calculate Haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(self.data)

        # Take the mean of the Haralick texture features
        ht_mean = np.mean(textures, axis=0)

        return ht_mean


class ImageGenerator:
    """
    A class for generating and preprocessing image data for COVID-19 detection.
    """

    def generate_from(self, path: str, divide: bool = False, reshape: bool = False, only_data: bool = False):
        """
        Generates an Image object from a given path.

        Parameters:
        ----------
        path : str
            Required. The path of the image file.
        divide : bool
            Optional. Default is False. Whether to divide the image into its RGB channels.
        reshape : bool
            Optional. Default is False. Whether to reshape the image into a specified shape.
        only_data : bool
            Optional. Default is False. Whether to return only the data of the image or the entire Image object.

        Returns:
        -------
        Image object or ndarray
            If only_data is True, returns the data of the image. Otherwise, returns the entire Image object.

        """
        image_files = glob(path + "/*g")
        for image_file in image_files:
            if only_data:
                yield Image(image_file, divide, reshape).data
            else:
                yield Image(image_file, divide, reshape)

    def generate_preprocessing_data(self, cov_path: str, cov_masks_path: str, non_cov_path: str, non_cov_masks_path: str) -> List:
        """
        Generates preprocessing data from four input paths.

        Parameters:
        ----------
        cov_path : str
            Required. The path of the COVID-19 image data.
        cov_masks_path : str
            Required. The path of the COVID-19 mask data.
        non_cov_path : str
            Required. The path of the non-COVID-19 image data.
        non_cov_masks_path : str
            Required. The path of the non-COVID-19 mask data.

        Returns:
        -------
        List
            A list containing the images and masks for COVID-19 and non-COVID-19 cases.

        """
        with ThreadPoolExecutor() as executor:
            cov_images = executor.submit(self.generate_from, cov_path)
            cov_masks = executor.submit(self.generate_from, cov_masks_path)
            non_cov_images = executor.submit(self.generate_from, non_cov_path)
            non_cov_masks = executor.submit(self.generate_from, non_cov_masks_path)

            return [cov_images.result(), cov_masks.result(), non_cov_images.result(), non_cov_masks.result()]

    def generate_classificator_data(self, cov_path, non_cov_path, divide=True, reshape=False):
        """
        Generate classification data from two paths, one for COVID-19 positive images and one for COVID-19 negative images. 

        Args:
            cov_path (str): Path to the folder containing COVID-19 positive images.
            non_cov_path (str): Path to the folder containing COVID-19 negative images.
            divide (bool, optional): Whether to divide the image into its RGB channels. Defaults to True.
            reshape (bool, optional): Whether to reshape the image into a specified shape. Defaults to False.

        Returns:
            tuple: A tuple containing the training and testing entries and results.

        The function uses ThreadPoolExecutor to generate the images from each path asynchronously. The positive and negative images are concatenated together and repeated 3 times. A results array is created with the length of the sum of both image sets, with 1s in the first cov_len entries. Finally, it returns a train_test_split of the entries and results with a test size of 0.2 and random state 0.
        """
        with ThreadPoolExecutor() as executor:
            cov_images = executor.map(lambda x: Image(x, divide, reshape).data, glob(cov_path + "/*g"))
            non_cov_images = executor.map(lambda x: Image(x, divide, reshape).data, glob(non_cov_path + "/*g"))

        entries = np.concatenate((list(cov_images), list(non_cov_images)))
        entries = np.repeat(entries[..., np.newaxis], 3, -1)

        cov_len = len(list(cov_images))
        non_cov_len = len(list(non_cov_images))
        results_len = cov_len + non_cov_len
        results = np.zeros((results_len))

        results[0:cov_len] = 1

        # Split into train and test
        return train_test_split(
            entries, results, test_size=0.2, random_state=0)

    def generate_processed_data(self, cov_processed_path, non_cov_processed_path, divide=True, reshape=False):
        """Generate processed data from two paths.

        Args:
            cov_processed_path (str): The path for processed COVID-19 images.
            non_cov_processed_path (str): The path for processed non-COVID-19 images.
            divide (bool, optional): Whether to divide the image into its RGB channels. Defaults to True.
            reshape (bool, optional): Whether to reshape the image into a specified shape. Defaults to False.

        Returns:
            list: A list containing the cov_images and non_cov_images.

        This method generates processed data from two paths, `cov_processed_path` and `non_cov_processed_path`. 
        It uses `ThreadPoolExecutor` to submit the `generate_from` function with each path, passing the `divide` 
        and `reshape` parameters. The method returns a list containing the `cov_images` and `non_cov_images`.
        """
        with ThreadPoolExecutor() as executor:
            cov_images = executor.submit(self.generate_from, cov_processed_path, divide, reshape)
            non_cov_images = executor.submit(self.generate_from, non_cov_processed_path, divide, reshape)

            return [cov_images.result(), non_cov_images.result()]


class ImageSaver:
    def __init__(self, images):
        """
        Create an ImageSaver object with a list of Image objects to save.

        Args:
            images (list[Image]): A list of Image objects to save.
        """
        self.images = images

    def save_to(self, path_dir):
        """
        Save all Image objects in the ImageSaver object to the given directory.

        Args:
            path_dir (str): The directory to save the images in.
        """
        for img in self.images:
            img.save_to(path_dir)


class ImageProcessor:

    def __init__(self, images, masks):
        """
        Initialize ImageProcessor class with images and masks.

        Args:
        - images (list): list of image objects
        - masks (list): list of mask objects
        """
        self.images = images
        self.masks = masks

    @staticmethod
    @njit(parallel=True)
    def __apply_mask(img, mask):
        """
        Apply mask to an image.

        Args:
        - img (ndarray): 2D array of shape (512, 512)
        - mask (ndarray): 2D array of shape (512, 512)

        Returns:
        - modified_img (ndarray): modified 2D array of shape (512, 512)
        """
        modified_img = np.copy(img)
        for i in prange(img.shape[0]):
            for j in prange(img.shape[1]):
                if mask[i, j] <= 20:
                    modified_img[i, j] = 0
        return modified_img

    def __process_image(self, args):
        """
        Process image with histogram equalization and mask.

        Args:
        - args (tuple): a tuple of (image, mask)

        Returns:
        - processed_image (ndarray): processed 2D array of shape (512, 512)
        """
        img, mask = args
        img_data = cv2.equalizeHist(img.data)
        processed_image = self.__apply_mask(img_data, mask.data)
        return processed_image

    def process(self):
        """
        Process all images and masks.

        Returns:
        - processed_images (list): list of processed image objects
        """
        iterables = np.swapaxes([self.images, self.masks], 0, 1)
        processed_images = []
        for args in tqdm(iterables):
            processed_image = self.__process_image(args)
            processed_images.append(processed_image)
        return processed_images


class ImageSegmenter:
    """
    Class for segmenting lung images using the U-Net model.

    This code is based on the work presented in:
    https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen/execution#4.-Results
    """

    def __init__(self, input_size=(512, 512, 1),
                 target_size=(512, 512),
                 folder_in='',
                 folder_out=''):
        """
        Initializes an ImageSegmenter object.

        Args:
        - input_size: a tuple representing the input shape of the U-Net model.
        - target_size: a tuple representing the target shape of the input images.
        - folder_in: a string representing the path to the input folder containing the lung images.
        - folder_out: a string representing the path to the output folder where the segmented images will be saved.
        """
        self.input_size = input_size
        self.target_size = target_size
        self.folder_in = folder_in
        self.folder_out = folder_out

    def __load_image(self, test_file):
        """
        Loads and preprocesses an image file.

        Args:
        - test_file: a string representing the path to the image file.

        Returns:
        - A numpy array representing the preprocessed image.
        """
        img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
        img = img / 255
        img = cv2.resize(img, self.target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        return img

    def __generator(self, test_files):
        """
        Generator function that yields preprocessed image arrays.

        Args:
        - test_files: a list of strings representing the paths to the image files.

        Yields:
        - A numpy array representing the preprocessed image.
        """
        for test_file in test_files:
            yield self.__load_image(test_file)

    def __save_result(self, save_path, npyfile, test_files):
        """
        Saves the segmented images to disk.

        Args:
        - save_path: a string representing the path to the output folder.
        - npyfile: a numpy array representing the segmented images.
        - test_files: a list of strings representing the paths to the input image files.
        """
        for i, item in enumerate(npyfile):
            result_file = test_files[i]
            img = (item[:, :, 0] * 255.).astype(np.uint8)

            filename, fileext = os.path.splitext(os.path.basename(result_file))

            result_file = os.path.join(save_path, "%s_predict%s" % (filename, fileext))

            cv2.imwrite(result_file, img)

    def segmentate(self):
        """
        This method loads the saved model from disk, generates image predictions for the images in the specified input folder, and saves the segmented images to the specified output folder.

        :return: None
        """
        # Load saved model from disk
        model = unet_model(input_size=self.input_size)
        model.load_weights('segmentation_model.hdf5')

        # Get list of image files from input folder
        files = glob(self.folder_in + "/*g")

        # Generate predictions for images in input folder
        gen = self.__generator(files)
        results = model.predict_generator(gen, len(files), verbose=1)

        # Save segmented images to output folder
        self.__save_result(self.folder_out, results, files)


class ImageCharacteristics:
    def __init__(self, cov_images, non_cov_images):
        """
        Initializes an ImageCharacteristics object with a list of covered and non-covered images.

        Args:
        - cov_images (list): A list of Image objects representing the covered images.
        - non_cov_images (list): A list of Image objects representing the non-covered images.
        """
        self.cov_images = cov_images
        self.non_cov_images = non_cov_images

    def save(self, file_path):
        """
        Computes the histogram and haralick features for each image and saves them to a csv file.

        Args:
        - file_path (str): The path to the file where the data will be saved.
        """
        # Compute the histogram and haralick features for each non-covered image
        non_cov_data = [np.hstack((img.hist(), img.haralick(), [0]))
                        for img in self.non_cov_images]

        # Compute the histogram and haralick features for each covered image
        cov_data = [np.hstack((img.hist(), img.haralick(), [1]))
                    for img in self.cov_images]

        # Combine the data for covered and non-covered images
        data = non_cov_data + cov_data

        # Convert the data to a pandas DataFrame and save it to a csv file
        pd.DataFrame(data).to_csv(file_path, index=False)


class ImageDataHistogram:
    @staticmethod
    def __hist_mean(images):
        """
        This function takes a list of images as an argument and returns the mean of the histograms of each image.
        It first creates a list of histograms for each image using the img.hist() method, then calculates the mean of all
        the histograms using the np.mean() method with axis=0 indicating that the mean should be calculated along the columns.
        Finally, it returns the mean value.
        """
        histograms = [img.hist() for img in images]
        hist_mean = np.mean(histograms, axis=0)
        return hist_mean

    @staticmethod
    def hist_mean(path):
        """
        This function takes in a path and returns the mean of the histogram of the image at that path.
        It generates an image using the ImageGenerator class and calls ImageDataHistogram's __hist_mean() method to calculate
        the mean of the histogram.
        """
        return ImageDataHistogram.__hist_mean(ImageGenerator().generate_from(
            abs_path(path)))

    @staticmethod
    def __hist_median(images):
        """
        This function takes in a list of images (images) and returns the median of the histograms of each image (hist_median).
        It uses a list comprehension to create a list of the histograms for each image (histograms), then uses NumPy's median()
        function to calculate the median of those histograms, with axis=0 indicating that the median should be calculated along
        the first axis. The result is returned as hist_median.
        """
        histograms = [img.hist() for img in images]
        hist_median = np.median(histograms, axis=0)
        return hist_median

    @staticmethod
    def hist_median(path):
        """
        This function takes in a path to an image and returns the median of its histogram.
        It generates an image using the ImageGenerator class and calls the __hist_median() method of the ImageDataHistogram class
        to get the median of its histogram.
        """
        return ImageDataHistogram.__hist_median(ImageGenerator().generate_from(
            abs_path(path)))
