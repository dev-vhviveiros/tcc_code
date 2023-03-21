from image import ImageSegmentator, ImageGenerator, ImageProcessor, ImageSaver, ImageCharacteristics
from utils import check_folder, abs_path
from wandb_utils import WandbUtils
from wb_dataset_representation import WBCovidDatasetArtifact, WBCovidMaskDatasetArtifact, WBCovidProcessedDatasetArtifact, WBNormalDatasetArtifact, WBNormalMaskDatasetArtifact, WBNormalProcessedDatasetArtifact


class Preprocessing:
    def __init__(self, wandb: WandbUtils):
        self.covid_path = abs_path('dataset/covid')
        self.covid_masks_path = abs_path('cov_masks')

        self.normal_path = abs_path('dataset/normal')
        self.normal_masks_path = abs_path('non_cov_masks')
        self.wandb = wandb

    def segment_lungs(self):
        covid_artifact = self.wandb.load_dataset_artifact(WBCovidDatasetArtifact())
        normal_artifact = self.wandb.load_dataset_artifact(WBNormalDatasetArtifact())

        # For re-creating the folders
        check_folder(self.covid_masks_path)
        check_folder(self.normal_masks_path)

        ImageSegmentator(folder_in=covid_artifact, folder_out=self.covid_masks_path).segmentate()
        ImageSegmentator(folder_in=normal_artifact, folder_out=self.normal_masks_path).segmentate()

        self.wandb.upload_dataset_artifact(WBCovidMaskDatasetArtifact())
        self.wandb.upload_dataset_artifact(WBNormalMaskDatasetArtifact())

    def process_images(self):
        generator = ImageGenerator()

        covid_images, covid_masks, non_covid_images, non_covid_masks = generator.generate_preprocessing_data(
            self.covid_path,
            self.covid_masks_path,
            self.normal_path,
            self.normal_masks_path
        )

        cov_processor = ImageProcessor(list(covid_images.result()), list(covid_masks.result()))
        non_cov_processor = ImageProcessor(list(non_covid_images.result()), list(non_covid_masks.result()))

        print("Processing images\n")
        cov_processed = cov_processor.process()
        non_cov_processed = non_cov_processor.process()

        # Saving processed images
        cov_save_path = abs_path('cov_processed')
        non_cov_save_path = abs_path('non_cov_processed')

        check_folder(cov_save_path)
        check_folder(non_cov_save_path)

        ImageSaver(cov_processed).save_to(cov_save_path)
        ImageSaver(non_cov_processed).save_to(non_cov_save_path)

        # Reading processed images
        generator = ImageGenerator()

        cov_processed_gen, non_cov_processed_gen = generator.generate_processed_data(
            self.covid_path, self.normal_path)

        cov_processed = list(cov_processed_gen.result())
        non_cov_processed = list(non_cov_processed_gen.result())

        # Saving characteristics
        characteristics_file = 'characteristics.csv'
        ic = ImageCharacteristics(cov_processed, non_cov_processed)
        ic.save(characteristics_file)

        # Saving histograms
        cov_histograms_path = abs_path('cov_histograms')
        non_cov_histograms_path = abs_path('non_cov_histograms')

        check_folder(cov_histograms_path)
        check_folder(non_cov_histograms_path)

        for i in cov_processed:
            i.save_hist(cov_histograms_path)

        for i in non_cov_processed:
            i.save_hist(non_cov_histograms_path)

        self.wandb.upload_dataset_artifact(WBCovidProcessedDatasetArtifact())
        self.wandb.upload_dataset_artifact(WBNormalProcessedDatasetArtifact())
