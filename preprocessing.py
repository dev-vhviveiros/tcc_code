from image import ImageSegmentator
from utils import check_folder, cov_path, covid_masks_path, normal_path, normal_masks_path
from wandb_utils import WandbUtils


class Preprocessing:
    def __init__(self, wandb: WandbUtils):
        self.covid_path = cov_path()
        self.covid_masks_path = covid_masks_path()

        self.normal_path = normal_path()
        self.normal_masks_path = normal_masks_path()
        self.wandb = wandb

    def segment_lungs(self):
        covid_artifact = self.wandb.load_covid_dataset_artifact()
        normal_artifact = self.wandb.load_normal_dataset_artifact()

        #For re-creating the folders
        check_folder(self.covid_masks_path)
        check_folder(self.normal_masks_path)

        #TODO: fix bug about the following appended string
        ImageSegmentator(folder_in=covid_artifact + "/covid",
                         folder_out=self.covid_masks_path).segmentate()
        ImageSegmentator(folder_in=normal_artifact + "/normal",
                         folder_out=self.normal_masks_path).segmentate()
        
        self.wandb.upload_covid_masks_dataset_artifact()
        self.wandb.upload_normal_masks_dataset_artifact()
