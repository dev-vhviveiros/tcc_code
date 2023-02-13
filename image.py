import cv2
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from models import unet_model
from tqdm import tqdm
from numba import prange, njit
import pandas as pd
import os
from utils import abs_path
from glob import glob
from matplotlib import pyplot as plt
import mahotas as mt


class Image:
    def __init__(self, file_path, divide=False, reshape=False):
        self.path = file_path
        self.divide = divide
        self.reshape = reshape
        self.data = self.__load_file()

    def __load_file(self, target_size=(512, 512)):
        """This function is used to load an image file. The image is read in grayscale using the cv2 library. If the divide parameter is true, the image is divided by 255. The image is then resized to the target size specified in the parameters. Finally, if the reshape parameter is true, the image is reshaped to a 1D array with an extra dimension of 1 added at the end. The function then returns the reshaped or unreshaped image depending on the value of reshape."""
        img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        if self.divide:
            img = img / 255
        img = cv2.resize(img, target_size)
        if self.reshape:
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
        return img

    def get_file_dir(self):
        return os.path.splitext(os.path.basename(self.path))

    def save_to(self, path_dir):
        """This function returns the directory of a file. The function takes in a "self" argument, which is an object containing the path of the file. It then uses Python's os module to split the path into two parts: the filename and its extension. Finally, it returns only the filename (without its extension) by accessing the first element of the tuple returned by os.path.splitext()."""
        filename, fileext = self.get_file_dir()
        result_file = abs_path(
            path_dir, "%s_processed%s" % (filename, fileext))
        cv2.imwrite(result_file, self.data)

    def shape(self):
        """This function returns the shape of the data stored in the object. The shape is a tuple of integers representing the size of each dimension of the data."""
        return self.data.shape

    def hist(self):
        """This code calculates a histogram of the data attribute of the object and returns it as an array. The calculation is done using OpenCV's calcHist() function, which takes in an array of images, an array of channels to be used, a mask, a range of bins and a range of values. The result is then squeezed into a single dimensional array and converted to an integer type before being returned."""
        result = np.squeeze(cv2.calcHist(
            [self.data], [0], None, [255], [1, 256]))
        result = np.asarray(result, dtype='int32')
        return result

    def save_hist(self, save_folder=''):
        """This function saves a histogram of the data stored in the object. It uses matplotlib's plt module to plot a histogram of the data and then saves it to a file in the specified save folder. The filename and file extension of the object are retrieved using the get_file_dir() method, and then used to create an absolute path for saving the result file. The histogram is calculated using OpenCV's calcHist() method, which takes in an array of values (self.data) and returns a histogram with 254 bins between 1 and 255."""
        plt.figure()
        histg = cv2.calcHist([self.data], [0], None, [254], [
            1, 255])  # calculating histogram
        plt.plot(histg)
        filename, fileext = self.get_file_dir()
        result_file = abs_path(
            save_folder, "%s_histogram%s" % (filename, '.png'))
        plt.savefig(result_file)
        plt.close()

    def haralick(self):
        """This function calculates Haralick texture features for 4 types of adjacency and then takes the mean of it and returns it. The function takes in an argument "self" which is assumed to be an object containing a data attribute. The Haralick texture features are calculated using the "mt.features.haralick" function, which takes in the data attribute as an argument. The mean of the textures is then calculated using the "mean" method, with the axis set to 0, and this mean is returned by the function."""
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(self.data)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean


class ImageGenerator:
    def generate_from(self, path, divide=False, reshape=False, only_data=False):
        """This function generates an Image object from a given path. 
Parameters: 
- path (required): the path of the image file
- divide (optional, default False): whether to divide the image into its RGB channels
- reshape (optional, default False): whether to reshape the image into a specified shape 
- only_data (optional, default False): whether to return only the data of the image or the entire Image object 
It uses glob() to get all images in the given path and iterates through them. For each image, it creates an Image object with specified parameters and yields either that object or its data depending on the value of only_data."""
        image_files = glob(path + "/*g")
        for image_file in image_files:
            if only_data:
                yield Image(image_file, divide, reshape).data
            else:
                yield Image(image_file, divide, reshape)

    def generate_preprocessing_data(self,
                                    cov_path,
                                    cov_masks_path,
                                    non_cov_path,
                                    non_cov_masks_path):
        """This function generates preprocessing data from four input paths. It uses a ThreadPoolExecutor to generate images and masks from each of the four paths and returns a list containing the images and masks."""
        with ThreadPoolExecutor() as executor:
            cov_images = executor.submit(self.generate_from, cov_path)
            cov_masks = executor.submit(self.generate_from, cov_masks_path)

            non_cov_images = executor.submit(
                self.generate_from, non_cov_path)
            non_cov_masks = executor.submit(
                self.generate_from, non_cov_masks_path)

            return [cov_images, cov_masks, non_cov_images, non_cov_masks]

    def generate_classificator_data(self, cov_path, non_cov_path, divide=True, reshape=False):
        """This function generates classificator data from two paths, one for cov images and one for non cov images. It uses a ThreadPoolExecutor to generate the images from each path. The cov_images and non_cov_images are then concatenated together and repeated 3 times. A results array is created with the length of the sum of both image sets, with 1s in the first cov_len entries. Finally, it returns a train_test_split of the entries and results with a test size of 0.2 and random state 0."""
        with ThreadPoolExecutor() as executor:
            cov_images = executor.submit(
                self.generate_from, cov_path, divide, reshape, True)

            non_cov_images = executor.submit(
                self.generate_from, non_cov_path, divide, reshape, True)

            cov_images = list(cov_images.result())
            non_cov_images = list(non_cov_images.result())

            entries = np.concatenate((cov_images, non_cov_images))
            entries = np.repeat(entries[..., np.newaxis], 3, -1)

            cov_len = len(cov_images)
            non_cov_len = len(non_cov_images)
            results_len = cov_len + non_cov_len
            results = np.zeros((results_len))

            results[0:cov_len] = 1

            # Split into test and training
            return train_test_split(
                entries, results, test_size=0.2, random_state=0)

    def generate_processed_data(self, cov_processed_path, non_cov_processed_path, divide=True, reshape=False):
        """This function generates processed data from two paths, cov_processed_path and non_cov_processed_path. It uses a ThreadPoolExecutor to submit the generate_from function with each path. The divide and reshape parameters are set to True and False respectively by default. The function returns a list containing the cov_images and non_cov_images."""
        with ThreadPoolExecutor() as executor:
            cov_images = executor.submit(
                self.generate_from, cov_processed_path)
            non_cov_images = executor.submit(
                self.generate_from, non_cov_processed_path)

            return [cov_images, non_cov_images]


class ImageSaver:
    def __init__(self, images):
        self.images = images

    def save_to(self, path_dir):
        for img in self.images:
            img.save_to(path_dir)


class ImageProcessor:

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    @staticmethod
    @njit()
    def __apply_mask(img, mask):
        """This code applies a mask to an image. 
        It takes two arguments, img and mask. The img argument is a 2D array of shape (512, 512). The mask argument is also a 2D array of shape (512, 512). The code uses prange to iterate over the rows and columns of the image. For each pixel in the image, if the corresponding pixel in the mask is less than or equal to 20, then the pixel in the image is set to 0. Finally, the modified image is returned."""
        imshape = img.shape

        for i in prange(0, imshape[0]):  # imshape[0] = 512
            for j in prange(0, imshape[1]):  # imshape[1] = 512
                if mask[i][j] <= 20:
                    img[i][j] = 0
        return img

    def __process_image(self, args):
        """This function takes in two arguments, an image and a mask, and processes the image. It first applies a histogram equalization to the image data, then applies the mask to the image data using the __apply_mask() method. Finally, it returns the processed image."""
        img, mask = args
        img.data = cv2.equalizeHist(img.data)
        img.data = ImageProcessor.__apply_mask(
            np.asarray(img.data), np.asarray(mask.data))
        return img

    def process(self):
        """This code creates an iterables variable which is the result of swapping the axes of self.images and self.masks. It then creates a total_size variable which is the length of iterables. It then creates an empty list called list. It then uses a for loop to loop through each item in iterables using tqdm and prange, and appends the result of calling __process_image on each item to list. Finally, it returns list."""
        iterables = np.swapaxes([self.images, self.masks], 0, 1)
        total_size = len(iterables)
        list = []
        for i in tqdm(prange(total_size)):
            list.append(self.__process_image(iterables[i]))
        return list


class ImageSegmentator:
    """
    Code using the model of the work accessed in:
    https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen/execution#4.-Results
    """

    def __init__(self, input_size=(512, 512, 1),
                 target_size=(512, 512),
                 folder_in='',
                 folder_out=''):
        self.input_size = input_size
        self.target_size = target_size
        self.folder_in = folder_in
        self.folder_out = folder_out

    def __test_load_image(self, test_file):
        img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
        img = img / 255
        img = cv2.resize(img, self.target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        return img

    def __test_generator(self, test_files):
        for test_file in test_files:
            yield self.__test_load_image(test_file)

    def __save_result(self, save_path, npyfile, test_files):
        for i, item in enumerate(npyfile):
            result_file = test_files[i]
            img = (item[:, :, 0] * 255.).astype(np.uint8)

            filename, fileext = os.path.splitext(os.path.basename(result_file))

            result_file = abs_path(
                save_path, "%s_predict%s" % (filename, fileext))

            cv2.imwrite(result_file, img)

    def segmentate(self):
        model = unet_model(input_size=self.input_size)
        model.load_weights('segmentation_model.hdf5')

        test_files = glob(self.folder_in + "/*g")

        test_gen = self.__test_generator(
            test_files)
        results = model.predict_generator(test_gen, len(test_files), verbose=1)
        self.__save_result(self.folder_out, results, test_files)


class ImageCharacteristics:
    def __init__(self, cov_images, non_cov_images):
        self.cov_images = cov_images
        self.non_cov_images = non_cov_images

    def save(self, file_path):
        """This code creates a pandas DataFrame from a list of images, and saves it as a csv file at the specified file path. The DataFrame contains the histogram and haralick features of each image, as well as an additional column indicating whether or not the image is covered (1 for covered, 0 for non-covered)."""
        # Histogram
        data = [np.hstack((img.hist(), img.haralick(), [0]))
                for img in self.non_cov_images]
        data += [np.hstack((img.hist(), img.haralick(), [1]))
                 for img in self.cov_images]

        pd.DataFrame(data).to_csv(file_path)


class ImageDataHistogram:
    @staticmethod
    def __hist_mean(images):
        """This function takes a list of images as an argument and returns the mean of the histograms of each image. The function first creates a list of histograms for each image using the img.hist() method. Then it calculates the mean of all the histograms using the np.mean() method, with axis=0 indicating that the mean should be calculated along the columns. Finally, it returns this mean value."""
        histograms = [img.hist() for img in images]
        hist_mean = np.mean(histograms, axis=0)
        return hist_mean

    @staticmethod
    def hist_mean(path):
        """This function takes in a path and returns the mean of the histogram of the image at that path. The function uses ImageGenerator() to generate an image from the given path and then calls ImageDataHistogram's __hist_mean() method to calculate the mean of the histogram."""
        return ImageDataHistogram.__hist_mean(ImageGenerator().generate_from(
            abs_path(path)))

    @staticmethod
    def __hist_median(images):
        """This function takes in a list of images (images) and returns the median of the histograms of each image (hist_median). The function uses a list comprehension to create a list of the histograms for each image (histograms), then uses NumPy's median() function to calculate the median of those histograms, with axis=0 indicating that the median should be calculated along the first axis. The result is returned as hist_median."""
        histograms = [img.hist() for img in images]
        hist_median = np.median(histograms, axis=0)
        return hist_median

    @staticmethod
    def hist_median(path):
        """This function takes in a path to an image and returns the median of its histogram. It does this by first generating an ImageGenerator object from the given path, then using the __hist_median() method of the ImageDataHistogram class to get the median of its histogram."""
        return ImageDataHistogram.__hist_median(ImageGenerator().generate_from(
            abs_path(path)))
