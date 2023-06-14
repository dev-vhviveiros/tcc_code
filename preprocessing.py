from image import LungMaskGenerator, ImageLoader, ImageProcessor, ImageSaver, ImageCharacteristics
from utils import check_folder, abs_path, images, cov_masks_path, cov_processed_path, normal_images, normal_masks_path, normal_processed_path, characteristics_path
from wandb_utils import WandbUtils
from wb_dataset_representation import WBCovidMaskDatasetArtifact, WBCovidProcessedDatasetArtifact, WBNormalMaskDatasetArtifact, WBNormalProcessedDatasetArtifact


class Preprocessing:
    def __init__(self, wandb: WandbUtils):
        """
        Initialize the Preprocessing object.

        Args:
            wandb (WandbUtils): The WandbUtils object to use for logging.
        """
        self.covid_path = images()
        self.covid_masks_path = cov_masks_path()
        self.normal_path = normal_images()
        self.normal_masks_path = normal_masks_path()
        self.characteristics_path = characteristics_path()
        self.wandb = wandb

    def generate_lungs_masks(self, covid_artifact, normal_artifact):
        """
        Generate lung masks for the COVID and normal chest X-ray images.

        Args:
            covid_artifact (wandb.Artifact): The COVID chest X-ray images artifact.
            normal_artifact (wandb.Artifact): The normal chest X-ray images artifact.

        Returns:
            None
        """
        # For re-creating the folders
        check_folder(self.covid_masks_path)
        check_folder(self.normal_masks_path)

        # Generate masks
        LungMaskGenerator(folder_in=covid_artifact, folder_out=self.covid_masks_path).generate()
        LungMaskGenerator(folder_in=normal_artifact, folder_out=self.normal_masks_path).generate()

        # Upload the artifacts
        self.wandb.upload_dataset_artifact(WBCovidMaskDatasetArtifact())
        self.wandb.upload_dataset_artifact(WBNormalMaskDatasetArtifact())

    def process_images(self, *artifacts):
        """
        Process the COVID and normal images, and save the processed images to the specified paths.

        Args:
            *artifacts: The COVID and normal chest X-ray images artifacts and their corresponding mask artifacts.
        Returns:
            None
        """
        # Load the dataset artifacts from wandb
        covid_artifact = artifacts[0]
        covid_mask_artifact = artifacts[1]
        normal_artifact = artifacts[2]
        normal_mask_artifact = artifacts[3]

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

    def generate_characteristics(self, cov_processed_artifact, normal_processed_artifact):
        """
        Generate image characteristics for the processed COVID and normal chest X-ray images.

        Args:
            cov_processed_artifact (wandb.Artifact): The processed COVID chest X-ray images artifact.
            normal_processed_artifact (wandb.Artifact): The processed normal chest X-ray images artifact.

        Returns:
            None
        """
        ic = ImageCharacteristics(cov_processed_artifact, normal_processed_artifact)
        ic.save(self.characteristics_path)
        self.wandb.upload_characteristics()
