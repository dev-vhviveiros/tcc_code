from args import (covid_data_path, 
                   covid_masks_data_path, 
                   non_covid_data_path, 
                   non_covid_masks_data_path)
from image import ImageSegmentator
from utils import check_folder

class Preprocessing:
    def __init__(self):
        self.covid_path = covid_data_path
        self.covid_masks_path = covid_masks_data_path

        self.non_covid_path = non_covid_data_path
        self.non_covid_masks_path = non_covid_masks_data_path
    
    def segment_lungs(self):
        check_folder(self.covid_masks_path)
        check_folder(self.non_covid_masks_path)

        ImageSegmentator(folder_in=self.covid_path,
                         folder_out=self.covid_masks_path).segmentate()
        ImageSegmentator(folder_in=self.non_covid_path,
                         folder_out=self.non_covid_masks_path).segmentate()
