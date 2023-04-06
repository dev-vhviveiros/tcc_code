from image import Image, LungMaskGenerator, ImageLoader, ImageProcessor, ImageSaver, ImageCharacteristics
from utils import check_folder, abs_path, images, cov_masks_path, cov_processed_path, normal_images, normal_masks_path, normal_processed_path
from wandb_utils import WandbUtils
from wb_dataset_representation import WBCovidDatasetArtifact, WBCovidMaskDatasetArtifact, WBCovidProcessedDatasetArtifact, WBNormalDatasetArtifact, WBNormalMaskDatasetArtifact, WBNormalProcessedDatasetArtifact


class Preprocessing:
    def __init__(self, wandb: WandbUtils):
        self.covid_path = images()
        self.covid_masks_path = cov_masks_path()
        self.normal_path = normal_images()
        self.normal_masks_path = normal_masks_path()
        self.wandb = wandb

    def generate_lungs_masks(self):
        # Download the artifacts
        covid_artifact = self.wandb.load_dataset_artifact(WBCovidDatasetArtifact())
        normal_artifact = self.wandb.load_dataset_artifact(WBNormalDatasetArtifact())

        # For re-creating the folders
        check_folder(self.covid_masks_path)
        check_folder(self.normal_masks_path)

        # Generate masks
        LungMaskGenerator(folder_in=covid_artifact, folder_out=self.covid_masks_path).generate()
        LungMaskGenerator(folder_in=normal_artifact, folder_out=self.normal_masks_path).generate()

        # Upload the artifacts
        self.wandb.upload_dataset_artifact(WBCovidMaskDatasetArtifact())
        self.wandb.upload_dataset_artifact(WBNormalMaskDatasetArtifact())

    def process_images(self):
        """
        Process the COVID and normal images, and save the processed images to the specified paths.
        """
        # Load the dataset artifacts from wandb
        covid_artifact = self.wandb.load_dataset_artifact(WBCovidDatasetArtifact())
        covid_mask_artifact = self.wandb.load_dataset_artifact(WBCovidMaskDatasetArtifact())
        normal_artifact = self.wandb.load_dataset_artifact(WBNormalDatasetArtifact())
        normal_mask_artifact = self.wandb.load_dataset_artifact(WBNormalMaskDatasetArtifact())

        # Initialize the image processors with the dataset artifacts
        cov_processor = ImageProcessor(covid_artifact, covid_mask_artifact)
        normal_processor = ImageProcessor(normal_artifact, normal_mask_artifact)

        # Process the images
        print("Processing images\n")
        cov_processed = cov_processor.process()
        normal_processed = normal_processor.process()

        # Save the processed images
        cov_save_path = cov_processed_path()
        normal_save_path = normal_processed_path()

        # Create the save paths if they don't exist, and delete any previous
        check_folder(cov_processed_path())
        check_folder(normal_processed_path())

        # Save the processed images to the specified paths
        ImageSaver(cov_processed).save_to(cov_save_path)
        ImageSaver(normal_processed).save_to(normal_save_path)

        # Upload the processed datasets to wandb
        self.wandb.upload_dataset_artifact(WBCovidProcessedDatasetArtifact())
        self.wandb.upload_dataset_artifact(WBNormalProcessedDatasetArtifact())

    def todo():
        # TODO those should be in another place
        # Reading processed images
        generator = ImageLoader()

        cov_processed_gen, normal_processed_gen = generator.generate_processed_data(
            self.covid_path, self.normal_path)

        cov_processed = list(cov_processed_gen.result())
        normal_processed = list(normal_processed_gen.result())

        # Saving characteristics
        characteristics_file = 'characteristics.csv'
        ic = ImageCharacteristics(cov_processed, normal_processed)
        ic.save(characteristics_file)

        # Saving histograms
        cov_histograms_path = abs_path('cov_histograms')
        normal_histograms_path = abs_path('normal_histograms')

        check_folder(cov_histograms_path)
        check_folder(normal_histograms_path)

        for i in cov_processed:
            i.save_hist(cov_histograms_path)

        for i in normal_processed:
            i.save_hist(normal_histograms_path)
