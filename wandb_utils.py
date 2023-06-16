
import numpy as np
import wandb
from image import Image, ImageLoader
from utils import CHARACTERISTICS_TAG, COVID_TAG, DATASET_TAG, MODEL_TAG, abs_path, model_path, cov_processed_path, normal_processed_path, characteristics_path
from utils import cov_processed, cov_images, cov_masks
from utils import normal_processed, normal_images, normal_masks
from vhviv_tools.json import json

from wb_dataset_representation import WBDatasetArtifact, WBCharacteristicsArtifact

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
    def __init__(self, wdb_data_alias):
        config = json("config.json")["wandb"]
        self.project_owner = config["wb_project_owner"]
        self.project_name = config["wb_project_name"]
        self.project_path = config["wb_project_path"]
        self.wdb_alias = wdb_data_alias
        self._run = wandb.init(project=self.project_name)

    def finish(self):
        """
        Finish the current WandB run by calling the `finish` method of the `run` object. Must be called after finishing the run jobs.

        Returns:
            None
        """
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
        result = callback(self._run)
        print("JOB: <" + job_type + "> DONE!")
        return result

    def download_artifact(self, name, relative_path, alias='latest'):
        """This function downloads an artifact from the W&B API. 
        Parameters: 
                self: the object that contains the project name 
                name: the name of the artifact to be downloaded 
                relative_path: a relative path to where the artifact should be downloaded 
                alias (optional): a string that specifies which version of the artifact should be (defaults to 'latest') 
        The function uses the W&B API to get an instance of the specified artifact, then downloads it to specified relative path. Finally, it prints a message indicating that the download was successful."""
        api = wandb.Api()
        artifact = api.artifact('vhviveiros/' + self.project_name + '/' + name + ":" + alias)
        artifact.download(root=abs_path(relative_path))
        print("Artifact " + name + " downloaded")

    def __create_wandb_table(self, images, masks, processed, tag):
        """This function creates a wandb table. It takes four parameters: images, masks, processed, and tag. The number of images, masks, and processed images must be the same or an exception is raised.

        The function then creates a table with three columns: Filename, Image, and Processed. For each item in the data array (which contains the images, masks and processed images), it creates an image object for each item. It then converts the mask data so that all values greater than 0 are considered 1. Finally it logs the table to wandb with the tag provided as a parameter."""
        if (len(images) != len(masks) or len(images) != len(processed)):
            raise Exception(
                "The number of images, masks and processed images must be the same")
        data = np.asarray([images, masks, processed])
        table = wandb.Table(
            columns=["Filename", "Image", "Processed"])
        for i in data.T:
            img = Image(i[0])
            mask = Image(i[1])
            img_proc = Image(i[2])

            img_data = img.data
            mask_data = mask.data
            img_proc_data = img_proc.data

            # The mask data must be converted in a way that values greater than 0 are considered 1.
            mask_data[mask_data > 0] = 1

            wandb_img = wandb.Image(img_data, masks={
                "mask": {
                    "mask_data": np.asarray(mask_data),
                    "class_labels": {
                        1: "Mask",
                    }
                }
            })

            wandb_img_proc = wandb.Image(img_proc_data)
            table.add_data(img.file_path, wandb_img, wandb_img_proc)
        wandb.log({tag + " Table": table})

    def __get_wb_artifact_path(self, tag: str) -> str:
        """This function gets the path of a model artifact from the project. The function returns a string in the format 'project_path/artifact_model_tag:wdb_data_alias'."""
        return '%s/%s:%s' % (self.project_path, tag, self.wdb_alias)

    def log_table(self):
        """ The function takes in the self parameter, which is a reference to the current instance of the class. Inside the function, a callback function is defined that takes in a run parameter. This callback function calls two other functions, __create_wandb_table and finish(), which create an interactive table in W&B with covid and non-covid images and masks, respectively. Finally, the execute_with() method is called with the callback and job_log_table parameters."""
        def callback(run):
            self.__create_wandb_table(cov_images(), cov_masks(), cov_processed(), "covid")
            self.__create_wandb_table(normal_images(), normal_masks(), normal_processed(), "non-covid")

        self.run_job(callback, WB_JOB_LOG_TABLE)

    def log_histogram_chart_comparison(self):
        """This function creates a log of histogram chart comparison. It first creates two image generators from cov_processed_path and non_cov_processed_path. Then it creates two histogram data sets from the generated images, cov_hist_data and non_cov_hist_data. The data is then converted to a list of tuples and stored in cov_table and non_cov_table. Finally, the tables are logged in the run."""
        def callback(run):
            cov_processed_gen = ImageLoader().load_from(cov_processed_path())
            non_cov_processed_gen = ImageLoader().load_from(normal_processed_path())

            cov_hist_data = np.transpose([x.hist() for x in cov_processed_gen]).tolist()
            non_cov_hist_data = np.transpose([x.hist() for x in non_cov_processed_gen]).tolist()

            cov_hist_data = list(zip([*range(1, 256)], cov_hist_data))
            non_cov_hist_data = list(zip([*range(1, 256)], non_cov_hist_data))

            cov_hist_data = [list(x) for x in list(cov_hist_data)]
            non_cov_hist_data = [list(x) for x in list(non_cov_hist_data)]

            cov_table = wandb.Table(data=cov_hist_data, columns=["Intensity", "Value"])
            non_cov_table = wandb.Table(data=non_cov_hist_data, columns=["Intensity", "Value"])

            run.log({"cov_chart": cov_table})
            run.log({"non_cov_chart": non_cov_table})
        self.run_job(callback, WB_JOB_HISTOGRAM_CHART)

    def upload_dataset_artifact(self, dataset_artifact: WBDatasetArtifact):
        """This method is used to upload a dataset artifact to W&B using the provided WBDatasetArtifact object. It creates a W&B artifact with the tag and type specified in the WBDatasetArtifact object and adds the directory specified in the object to the artifact. It then logs the artifact to W&B using the provided aliases.

        Parameters:
            dataset_artifact (WBDatasetArtifact): An instance of WBDatasetArtifact class that encapsulates information about the dataset artifact to upload."""
        def callback(run):
            artifact = wandb.Artifact(dataset_artifact.tag, type=DATASET_TAG)
            artifact.add_dir(dataset_artifact.path)
            run.log_artifact(artifact, aliases=dataset_artifact.aliases + [self.wdb_alias])

        self.run_job(callback, WB_JOB_UPLOAD_DATASET)

    def load_dataset_artifact(self, dataset_artifact: WBDatasetArtifact) -> str:
        """Loads a previously uploaded dataset artifact and downloads it to the local machine.
            Parameters:
                dataset_artifact (WBDatasetArtifact): An instance of the WBDatasetArtifact class that specifies the artifact to be loaded.

            Returns:
                A string representing the local path where the artifact was downloaded to."""
        def callback(run):
            artifact_wdb_path = dataset_artifact.wb_artifact_path(self.project_path, self.wdb_alias)
            artifact = run.use_artifact(artifact_wdb_path, type=DATASET_TAG)
            return artifact.download()

        return self.run_job(callback, WB_JOB_LOAD_DATASET)

    def load_model_artifact(self, run) -> str:
        """This function loads a model from a run in W&B. 
        It takes in a parameter 'run' which is the run from which the model should be loaded. 
        The function first uses the W&B artifact associated with the run to download the model. It then returns the directory of the model."""
        artifact_wdb_path = self.__get_wb_artifact_path(MODEL_TAG)
        dataset_artifact = run.use_artifact(artifact_wdb_path, type=MODEL_TAG)
        model_dir = dataset_artifact.download()

        return model_dir

    def generate_model_artifact(self):
        """This code defines a function called generate_model_artifact that takes in an object of the class it is defined in as a parameter. The function creates an artifact object from the Wandb library, with the tag given by the self.artifact_model_tag parameter and type set to "model". It then adds a file located at model_path to the artifact and returns it."""
        model = wandb.Artifact(MODEL_TAG, type=MODEL_TAG)
        model.add_file(model_path())
        return model

    def upload_model_artifact(self, run):
        """This function uploads a model from a run in W&B. 
        It takes in a parameter 'run' which is the run from which the model should be uploaded. 
        The function first uses the W&B artifact associated with the run to upload the model. It then returns the directory of the model."""
        model_artifact = self.generate_model_artifact()
        run.log_artifact(model_artifact, aliases=[self.wdb_alias])

    def load_characteristics(self):
        """
        Downloads the characteristics artifact from the W&B run and returns it.

        Returns:
            The downloaded characteristics artifact.
        """
        def callback(run):
            artifact_wdb_path = WBCharacteristicsArtifact().wb_artifact_path(self.project_path, self.wdb_alias)
            artifact = run.use_artifact(artifact_wdb_path, type=CHARACTERISTICS_TAG)
            return artifact.download()

        return self.run_job(callback, WB_JOB_LOAD_DATASET)

    def upload_characteristics(self):
        """
        Uploads the characteristics file to the W&B run as an artifact.

        Returns:
            None.
        """
        def callback(run):
            artifact = wandb.Artifact(CHARACTERISTICS_TAG, type=CHARACTERISTICS_TAG)
            artifact.add_file(characteristics_path())
            run.log_artifact(artifact, aliases=[CHARACTERISTICS_TAG, self.wdb_alias])

        self.run_job(callback, WB_JOB_UPLOAD_DATASET)

    def log_trainset(self, training_set, testing_set):  # TODO
        def callback(run):
            raw_data = wandb.Artifact(
                COVID_TAG, type=DATASET_TAG,
                description="Raw covid dataset, split into train/test",
                metadata={"sizes": [len(dataset) for dataset in [training_set, testing_set]]})
            run.log_artifact(raw_data)

        self.run_job(callback)
