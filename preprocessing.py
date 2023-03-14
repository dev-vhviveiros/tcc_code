from image import ImageSegmentator
from utils import check_folder, cov_path, covid_masks_path, normal_path, normal_masks_path


class Preprocessing:
    def __init__(self):
        self.covid_path = cov_path
        self.covid_masks_path = covid_masks_path

        self.non_covid_path = normal_path
        self.non_covid_masks_path = normal_masks_path

    def segment_lungs(self):
        check_folder(self.covid_masks_path)
        check_folder(self.non_covid_masks_path)

        ImageSegmentator(folder_in=self.covid_path,
                         folder_out=self.covid_masks_path).segmentate()
        ImageSegmentator(folder_in=self.non_covid_path,
                         folder_out=self.non_covid_masks_path).segmentate()
