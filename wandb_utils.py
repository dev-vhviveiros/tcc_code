import numpy as np
import wandb
from image import Image, ImageLoader
from utils import abs_path, load_config

from dataset_representation import CHARACTERISTICS_TAG, COVID_TAG, DATASET_TAG, MODEL_TAG, CovidDataset, CovidMaskDataset, CovidProcessedDataset, DatasetRepresentation, Characteristics, Model, NormalDataset, NormalMaskDataset, NormalProcessedDataset

# Project variables (just to avoid repeating/misspelling them)
# JOB VARIABLES:
WB_JOB_UPLOAD_DATASET = "upload_dataset"
WB_JOB_LOAD_DATASET = "load_dataset"
WB_JOB_LOG_TABLE = "log_interactive_table"
WB_JOB_HISTOGRAM_CHART = "log_histogram_chart"
WB_JOB_LOAD_ARTIFACTS = "load_artifacts"
WB_JOB_MODEL_FIT = "model_fit"
WB_JOB_LOG_TRAINING_DATA = "log_training_data"


class WandbUtils:
    """ A utility class for interacting with the Weights and Biases (WandB) platform.
    Args:
        wdb_data_alias (str): The alias of the WandB data to use for this run.
    """

    def __init__(self, wdb_tag):
        """
        Initializes a new instance of the `WandbUtils` class.

        Args:
            wdb_data_alias (str): The alias of the WandB data to use for this run.
        """
        self.project_owner = load_config("wb_project_owner")
        self.project_name = load_config("wb_project_name")
        self.project_path = load_config("wb_project_path")
        self.wdb_tag = wdb_tag
        self._run = wandb.init(project=self.project_name, group="main", tags=wdb_tag)

    def finish(self):
        """
        Finish the current WandB run by calling the `finish` method of the `run` object.
        This method must be called at the end of each run to ensure that all data is properly logged.
        """
        # Call the `finish` method of the `run` object to finalize the run.
        self._run.finish()

    def run_job(self, callback, job_type) -> any:
        """
        Run a job with the given callback function and job type.

        Args:
            callback (function): The callback function to run.
            job_type (str): The type of job being run.

        Returns:
            any: The result of the callback function.
        """
        print(f"WDB RUNNING JOB: <{job_type}>!")
        # Call the callback function with the `run` object as an argument and store the result.
        result = callback(self._run)

        # Print a message indicating that the job is done.
        print(f"WDB JOB: <{job_type}> DONE!")

        # Return the result of the callback function.
        return result

    def download_artifact(self, name, relative_path, alias='latest'):
        """
        Download an artifact from the W&B API.

        Args:
            name (str): The name of the artifact to be downloaded.
            relative_path (str): A relative path to where the artifact should be downloaded.
            alias (str, optional): A string that specifies which version of the artifact should be downloaded (defaults to 'latest').

        Returns:
            None
        """
        # Create a new instance of the W&B API.
        api = wandb.Api()

        # Get an instance of the specified artifact using the W&B API.
        artifact = api.artifact('vhviveiros/' + self.project_name + '/' + name + ":" + alias)

        # Download the artifact to the specified relative path.
        artifact.download(root=abs_path(relative_path))

        # Print a message indicating that the download was successful.
        print("Artifact " + name + " downloaded")

    def __create_wandb_table(self, images, masks, processed, tag):
        """
        Create a W&B table with the given images, masks, processed images, and tag.

        Args:
            images (list): A list of image file paths.
            masks (list): A list of corresponding mask file paths.
            processed (list): A list of corresponding processed image file paths.
            tag (str): A string that specifies the tag for the W&B table.

        Returns:
            None
        """
        # Check that the number of images, masks, and processed images are the same.
        if len(images) != len(masks) or len(images) != len(processed):
            raise Exception("The number of images, masks, and processed images must be the same")

        # Create a numpy array with the images, masks, and processed images.
        data = np.asarray([images, masks, processed])

        # Create a new W&B table with the columns "Filename", "Image", and "Processed".
        table = wandb.Table(columns=["Filename", "Image", "Processed"])

        # Iterate over each row in the data array and add it to the W&B table.
        for i in data.T:
            # Create a W&B image object for the original image.
            img = Image(i[0])
            wandb_img = wandb.Image(img.data)

            # Create a W&B image object for the mask.
            mask = Image(i[1])
            mask_data = mask.data
            mask_data[mask_data > 0] = 1
            wandb_mask = wandb.Image(mask_data, masks={
                "mask": {
                    "mask_data": np.asarray(mask_data),
                    "class_labels": {
                        1: "Mask",
                    }
                }
            })

            # Create a W&B image object for the processed image.
            img_proc = Image(i[2])
            wandb_img_proc = wandb.Image(img_proc.data)

            # Add the row to the W&B table.
            table.add_data(img.file_path, wandb_img, wandb_img_proc)

        # Log the W&B table with the specified tag.
        wandb.log({f"{tag} Table": table})

    def __get_wb_artifact_path(self, tag: str) -> str:
        """
        Get the path of a W&B artifact from the project.

        Args:
            tag (str): A string that specifies the tag for the W&B artifact.

        Returns:
            str: A string in the format 'project_path/artifact_model_tag:wdb_data_alias'.
        """
        # Return a string in the format 'project_path/artifact_model_tag:wdb_data_alias'.
        return '%s/%s:%s' % (self.project_path, tag, self.wdb_tag)

    def log_table(self):
        """
        Log an interactive table in W&B with covid and non-covid images and masks.

        Args:
            None

        Returns:
            None
        """
        # Define a callback function that takes in a `run` parameter.
        def callback(run):
            # Create a W&B table with covid images and masks.
            self.__create_wandb_table(CovidDataset().images(), CovidMaskDataset().images(),
                                      CovidProcessedDataset().images(), "covid")

            # Create a W&B table with non-covid images and masks.
            self.__create_wandb_table(NormalDataset().images(), NormalMaskDataset().images(),
                                      NormalProcessedDataset().images(), "non-covid")

            # Finish the current W&B run.
            self.finish()

        # Call the `run_job` method with the callback function and the `WB_JOB_LOG_TABLE` job type.
        self.run_job(callback, WB_JOB_LOG_TABLE)

    def log_histogram_chart_comparison(self, samples_target_size):
        """
        Log a comparison of histogram charts for covid and non-covid images.

        Args:
            None

        Returns:
            None
        """
        # Define a callback function that takes in a `run` parameter.
        def callback(run):
            # Load the image data for the covid and non-covid images.
            cov_processed_gen = ImageLoader().load_from(CovidProcessedDataset().path, samples_target_size)
            non_cov_processed_gen = ImageLoader().load_from(NormalProcessedDataset().path, samples_target_size)

            # Define a helper function to generate histogram data for an image generator.
            def generate_histogram_data(image_gen):
                # Define the length of the histogram.
                HISTOGRAM_LENGTH = 254
                # Initialize the histogram data to all zeros.
                hist_data = [0] * HISTOGRAM_LENGTH
                # Loop over each image in the generator.
                for _, img in enumerate(image_gen):
                    # Generate the histogram for the image.
                    hist = img.hist()
                    # Add the histogram data to the overall histogram data.
                    hist_data += hist[:-1]
                # Convert the histogram data to a list of tuples where each tuple contains the intensity value and the corresponding histogram value.
                return [(i, val) for i, val in enumerate(hist_data)]

            # Generate histogram data for the covid and non-covid images.
            cov_hist_data = generate_histogram_data(cov_processed_gen)
            non_cov_hist_data = generate_histogram_data(non_cov_processed_gen)

            # Create wandb.Table objects to represent the histogram data.
            cov_table = wandb.Table(data=cov_hist_data, columns=["Intensity", "Value"])
            non_cov_table = wandb.Table(data=non_cov_hist_data, columns=["Intensity", "Value"])

            # Create line and scatter plots for the histogram data.
            cov_line_plot = wandb.plot.line(cov_table, x="Intensity", y="Value", title="Covid Line Plot")
            non_cov_line_plot = wandb.plot.line(non_cov_table, x="Intensity", y="Value", title="Non-Covid Line Plot")
            cov_scatter_plot = wandb.plot.scatter(cov_table, x="Intensity", y="Value", title="Covid Scatter Plot")
            non_cov_scatter_plot = wandb.plot.scatter(
                non_cov_table, x="Intensity", y="Value", title="Non-Covid Scatter Plot")

            # Log the histogram data and plots to the wandb run.
            run.log({
                "covid_histogram_data": cov_table,
                "non_covid_histogram_data": non_cov_table,
                "covid_line_plot": cov_line_plot,
                "non_covid_line_plot": non_cov_line_plot,
                "covid_scatter_plot": cov_scatter_plot,
                "non_covid_scatter_plot": non_cov_scatter_plot
            })

        # Call the `run_job` method with the callback function and the `WB_JOB_HISTOGRAM_CHART` job type.
        self.run_job(callback, WB_JOB_HISTOGRAM_CHART)

    def upload_dataset_artifact(self, dataset_artifact: DatasetRepresentation):
        """
        Upload a dataset artifact to W&B using the provided DatasetRepresentation object.

        Args:
            dataset_artifact (DatasetRepresentation): An instance of the DatasetRepresentation class that encapsulates information about the dataset artifact to upload.

        Returns:
            None
        """
        # Define a callback function that takes in a `run` parameter.
        def callback(run):
            # Create a new W&B artifact with the tag and type specified in the DatasetRepresentation object.
            artifact = wandb.Artifact(dataset_artifact.tag, type=DATASET_TAG)

            # Add the directory specified in the DatasetRepresentation object to the artifact.
            artifact.add_dir(dataset_artifact.path)

            # Log the artifact to W&B using the provided aliases.
            run.log_artifact(artifact, aliases=dataset_artifact.aliases + [self.wdb_tag])

            # Wait for the artifact logging
            artifact.wait()

        # Call the `run_job` method with the callback function and the `WB_JOB_UPLOAD_DATASET` job type.
        self.run_job(callback, WB_JOB_UPLOAD_DATASET)

    def load_dataset_artifact(self, dataset_artifact: DatasetRepresentation) -> str:
        """
        Loads a previously uploaded dataset artifact and downloads it to the local machine.

        Args:
            dataset_artifact (DatasetRepresentation): An instance of the DatasetRepresentation class that specifies the artifact to be loaded.

        Returns:
            str: A string representing the local path where the artifact was downloaded to.
        """
        # Define a callback function that takes in a `run` parameter.
        def callback(run):
            # Get the W&B artifact path for the specified dataset artifact.
            artifact_wdb_path = dataset_artifact.wb_artifact_path(self.project_path, self.wdb_tag)
            # Use the W&B artifact with the specified path and type.
            artifact = run.use_artifact(artifact_wdb_path, type=DATASET_TAG)
            # Download the artifact to the local machine and return the path where it was downloaded to.
            return artifact.download()

        # Call the `run_job` method with the callback function and the `WB_JOB_LOAD_DATASET` job type.
        return self.run_job(callback, WB_JOB_LOAD_DATASET)

    def load_model_artifact(self, run) -> str:
        """
        Load a model from a W&B run.

        Args:
            run (wandb.Run): An instance of the wandb.Run class that represents the W&B run to load the model from.

        Returns:
            str: A string representing the local path where the model was downloaded to.
        """
        # Get the W&B artifact path for the model artifact.
        artifact_wdb_path = self.__get_wb_artifact_path(MODEL_TAG)

        # Use the W&B artifact with the specified path and type.
        artifact = run.use_artifact(artifact_wdb_path, type=MODEL_TAG)

        # Download the artifact to the local machine and return the path where it was downloaded to.
        return artifact.download()

    def generate_model_artifact(self):
        """
        Generate a W&B artifact for the model.

        Args:
            None

        Returns:
            wandb.Artifact: An instance of the wandb.Artifact class that represents the model artifact.
        """
        # Create a new W&B artifact with the tag and type specified in the constants.
        model = wandb.Artifact(MODEL_TAG, type=MODEL_TAG)

        # Add the model file to the artifact.
        model.add_file(Model().path)

        # Return the model artifact.
        return model

    def upload_model_artifact(self, run):
        """
        Upload the model artifact to W&B using the provided run.

        Args:
            run (wandb.Run): An instance of the wandb.Run class that represents the W&B run to upload the model artifact to.

        Returns:
            None
        """
        # Generate a new W&B artifact for the model.
        model_artifact = self.generate_model_artifact()

        # Log the model artifact to W&B using the provided aliases.
        run.log_artifact(model_artifact, aliases=[self.wdb_tag])

        # Finish the current W&B run.
        self.finish()

    def load_characteristics(self):
        """
        Downloads the characteristics artifact from the W&B run and returns it.

        Returns:
            str: A string representing the local path where the characteristics artifact was downloaded to.
        """
        # Define a callback function that takes in a `run` parameter.
        def callback(run):
            # Get the W&B artifact path for the characteristics artifact.
            artifact_wdb_path = Characteristics().wb_artifact_path(self.project_path, self.wdb_tag)
            # Use the W&B artifact with the specified path and type.
            artifact = run.use_artifact(artifact_wdb_path, type=CHARACTERISTICS_TAG)
            # Download the artifact to the local machine and return the path where it was downloaded to.
            return artifact.download() + "/" + load_config("generated_csv_file")

        # Call the `run_job` method with the callback function and the `WB_JOB_LOAD_DATASET` job type.
        return self.run_job(callback, WB_JOB_LOAD_DATASET)

    def upload_characteristics(self):
        """
        Upload the characteristics file to W&B as an artifact.

        Args:
            None

        Returns:
            None
        """
        # Define a callback function that takes in a `run` parameter.
        def callback(run):
            # Create a new W&B artifact with the tag and type specified in the constants.
            artifact = wandb.Artifact(CHARACTERISTICS_TAG, type=CHARACTERISTICS_TAG)

            # Add the characteristics file to the artifact.
            artifact.add_file(Characteristics().path)

            # Log the artifact to W&B using the provided aliases.
            run.log_artifact(artifact, aliases=[CHARACTERISTICS_TAG, self.wdb_tag])

        # Call the `run_job` method with the callback function and the `WB_JOB_UPLOAD_DATASET` job type.
        self.run_job(callback, WB_JOB_UPLOAD_DATASET)
