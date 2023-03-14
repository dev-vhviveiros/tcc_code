from ast import alias
from typing import List
import numpy as np
import wandb
from image import Image, ImageGenerator
from utils import WB_ARTIFACT_COVID_PROCESSED_TAG, WB_ARTIFACT_COVID_TAG, WB_ARTIFACT_COVID_MASKS_TAG, WB_ARTIFACT_DATASET_TAG, WB_ARTIFACT_MODEL_TAG, WB_ARTIFACT_NORMAL_MASKS_TAG, WB_ARTIFACT_NORMAL_PROCESSED_TAG, WB_ARTIFACT_NORMAL_TAG, WB_JOB_HISTOGRAM_CHART, WB_JOB_LOAD_DATASET, WB_JOB_LOG_TABLE, WB_JOB_UPLOAD_DATASET, abs_path, cov_path, covid_masks_path, dataset_path, model_path, cov_processed_path, normal_masks_path, normal_path, normal_processed_path
from utils import cov_processed, cov_images, cov_masks
from utils import normal_processed, normal_images, normal_masks
from vhviv_tools.json import json


class WandbUtils:
    def __init__(self, wdb_data_alias):
        config = json("config.json")["wandb"]
        self.project_owner = config["wb_project_owner"]
        self.project_name = config["wb_project_name"]
        self.project_path = config["wb_project_path"]
        self.wdb_alias = wdb_data_alias

    def execute_with(self, callback, job_type):
        """This code takes two parameters, self and callback. It then initializes a run with the project name and job type specified in the parameters. The callback function is then called with the run as an argument. Finally, it prints a message indicating that the job is done."""
        with (wandb.init(project=self.project_name, job_type=job_type)) as run:
            callback(run)

        print("JOB: <" + job_type + "> DONE!")

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
            table.add_data(img.path, wandb_img, wandb_img_proc)
        wandb.log({tag + " Table": table})

    def __get_wb_artifact_path(self, tag: str) -> str:
        """This function gets the path of a model artifact from the project. The function returns a string in the format 'project_path/artifact_model_tag:wdb_data_alias'."""
        return '%s/%s:%s' % (self.project_path, tag, self.wdb_alias)

    def log_interactive_table(self):
        """ The function takes in the self parameter, which is a reference to the current instance of the class. Inside the function, a callback function is defined that takes in a run parameter. This callback function calls two other functions, __create_wandb_table and finish(), which create an interactive table in W&B with covid and non-covid images and masks, respectively. Finally, the execute_with() method is called with the callback and job_log_table parameters."""
        def callback(run):
            self.__create_wandb_table(cov_images(), cov_masks(), cov_processed(), "covid")
            self.__create_wandb_table(normal_images(), normal_masks(), normal_processed(), "non-covid")
            run.finish()

        self.execute_with(callback, WB_JOB_LOG_TABLE)

    def log_histogram_chart_comparison(self):
        """This function creates a log of histogram chart comparison. It first creates two image generators from cov_processed_path and non_cov_processed_path. Then it creates two histogram data sets from the generated images, cov_hist_data and non_cov_hist_data. The data is then converted to a list of tuples and stored in cov_table and non_cov_table. Finally, the tables are logged in the run."""
        def callback(run):
            cov_processed_gen = ImageGenerator().generate_from(cov_processed_path())
            non_cov_processed_gen = ImageGenerator().generate_from(normal_processed_path())

            cov_hist_data = np.transpose([x.hist() for x in cov_processed_gen]).tolist()
            non_cov_hist_data = np.transpose([x.hist() for x in non_cov_processed_gen]).tolist()

            cov_hist_data = list(zip([*range(1, 256)], cov_hist_data))
            non_cov_hist_data = list(zip([*range(1, 256)], non_cov_hist_data))

            cov_hist_data = [list(x) for x in list(cov_hist_data)]
            non_cov_hist_data = [list(x) for x in list(non_cov_hist_data)]

            cov_table = wandb.Table(data=cov_hist_data, columns=["Intensity", "Value"])
            non_cov_table = wandb.Table(data=non_cov_hist_data, columns=["Intensity", "Value"])

            # fields = {"x": "Intensity", "value": "Value"}

            # cov_chart = wandb.plot_table(vega_spec_name="tcc/cov_chart",
            #                              data_table=cov_table,
            #                              fields=fields)
            # non_cov_chart = wandb.plot_table(vega_spec_name="tcc/non_cov_chart",
            #                                  data_table=non_cov_table,
            #                                  fields=fields)

            run.log({"cov_chart": cov_table})
            run.log({"non_cov_chart": non_cov_table})
        self.execute_with(callback, WB_JOB_HISTOGRAM_CHART)

    def upload_dataset_artifact(self, tag: str, job: str, dir_path: str, aliases: List[str]):
        """This function uploads a dataset artifact to W&B. 
            Parameters: 
                tag: a string that is used to identify the dataset artifact 
                job: a string that identifies the job associated with the dataset artifact 
                dir_path: a string that specifies the directory path of the dataset artifact 
                aliases: a list of strings that are used as aliases for the dataset artifact
            The function first creates an Artifact object with the given tag and type. It then adds the directory specified by dir_path to the Artifact and logs it with any given aliases. Finally, it executes a callback function which logs the Artifact to W&B."""
        def callback(run):
            artifact = wandb.Artifact(tag, type=WB_ARTIFACT_DATASET_TAG)
            artifact.add_dir(dir_path, tag)
            run.log_artifact(artifact, aliases=aliases)

        self.execute_with(callback, job)

    def upload_covid_dataset_artifact(self, aliases=[]):
        """This function uploads a COVID dataset artifact to the WDB and returns the result of the upload_dataset_artifact() function. Aliases params are only intended for extra tagging, so theyre not required."""
        return self.upload_dataset_artifact(WB_ARTIFACT_COVID_TAG, WB_JOB_UPLOAD_DATASET, cov_path(), [self.wdb_alias, WB_ARTIFACT_COVID_TAG] + aliases)

    def upload_covid_masks_dataset_artifact(self, aliases=[]):
        """This function uploads a COVID MASKS dataset artifact to the WDB and returns the result of the upload_dataset_artifact() function. Aliases params are only intended for extra tagging, so theyre not required."""
        return self.upload_dataset_artifact(WB_ARTIFACT_COVID_MASKS_TAG, WB_JOB_UPLOAD_DATASET, covid_masks_path(), [self.wdb_alias, WB_ARTIFACT_COVID_MASKS_TAG] + aliases)

    def upload_covid_processed_dataset_artifact(self, aliases=[]):
        """This function uploads a COVID PROCESSED dataset artifact to the WDB and returns the result of the upload_dataset_artifact() function. Aliases params are only intended for extra tagging, so theyre not required."""
        return self.upload_dataset_artifact(WB_ARTIFACT_COVID_PROCESSED_TAG, WB_JOB_UPLOAD_DATASET, cov_processed_path(), [self.wdb_alias, WB_ARTIFACT_COVID_PROCESSED_TAG] + aliases)

    def upload_normal_dataset_artifact(self, aliases=[]):
        """This function uploads a NORMAL dataset artifact to the WDB and returns the result of the upload_dataset_artifact() function. Aliases params are only intended for extra tagging, so theyre not required."""
        return self.upload_dataset_artifact(WB_ARTIFACT_NORMAL_TAG, WB_JOB_UPLOAD_DATASET, normal_path(), [self.wdb_alias, WB_ARTIFACT_NORMAL_TAG] + aliases)

    def upload_normal_masks_dataset_artifact(self, aliases=[]):
        """This function uploads a NORMAL MASKS dataset artifact to the WDB and returns the result of the upload_dataset_artifact() function. Aliases params are only intended for extra tagging, so theyre not required."""
        return self.upload_dataset_artifact(WB_ARTIFACT_NORMAL_MASKS_TAG, WB_JOB_UPLOAD_DATASET, normal_masks_path(), [self.wdb_alias, WB_ARTIFACT_NORMAL_MASKS_TAG] + aliases)

    def upload_normal_processed_dataset_artifact(self, aliases=[]):
        """This function uploads a NORMAL PROCESSED dataset artifact to the WDB and returns the result of the upload_dataset_artifact() function. Aliases params are only intended for extra tagging, so theyre not required."""
        return self.upload_dataset_artifact(WB_ARTIFACT_NORMAL_PROCESSED_TAG, WB_JOB_UPLOAD_DATASET, normal_processed_path(), [self.wdb_alias, WB_ARTIFACT_NORMAL_PROCESSED_TAG] + aliases)

    def load_dir_artifact(self, tag: str, job: str):
        """This function loads a directory artifact from a given job. 
            It takes two parameters: 
                tag (str): the tag of the artifact 
                job (str): the name of the job to load the artifact from 
            It then gets the path of the artifact and uses it to create an artifact object. 
            Finally, it downloads the artifact and returns it."""
        def callback(run):
            artifact_wdb_path = self.__get_wb_artifact_path(tag)
            dataset_artifact = run.use_artifact(artifact_wdb_path, type=WB_ARTIFACT_DATASET_TAG)
            return dataset_artifact.download()

        return self.execute_with(callback, job)

    def load_covid_dataset_artifact(self):
        """This function is used to load an artifact associated with the Covid-19 dataset. It takes in a self argument and returns a directory artifact associated with the Covid-19 tag and the job of loading the dataset."""
        return self.load_dir_artifact(WB_ARTIFACT_COVID_TAG, WB_JOB_LOAD_DATASET)

    def load_covid_masks_dataset_artifact(self):
        """This function is used to load an artifact associated with the Covid-19 masks dataset. It takes in a self argument and returns a directory artifact associated with the Covid-19 masks tag and the job of loading the dataset."""
        return self.load_dir_artifact(WB_ARTIFACT_COVID_MASKS_TAG, WB_JOB_LOAD_DATASET)
    
    def load_covid_processed_dataset_artifact(self):
        """This function is used to load an artifact associated with the Covid-19 processed dataset. It takes in a self argument and returns a directory artifact associated with the Covid-19 processed tag and the job of loading the dataset."""
        return self.load_dir_artifact(WB_ARTIFACT_COVID_PROCESSED_TAG, WB_JOB_LOAD_DATASET)

    def load_normal_dataset_artifact(self):
        """This function is used to load a normal dataset artifact from the directory WB_ARTIFACT_NORMAL_TAG. The artifact is loaded using the job WB_JOB_LOAD_DATASET."""
        return self.load_dir_artifact(WB_ARTIFACT_NORMAL_TAG, WB_JOB_LOAD_DATASET)

    def load_normal_masks_dataset_artifact(self):
        """This function is used to load an artifact associated with the normal masks dataset. It takes in a self argument and returns a directory artifact associated with the normal masks tag and the job of loading the dataset."""
        return self.load_dir_artifact(WB_ARTIFACT_NORMAL_MASKS_TAG, WB_JOB_LOAD_DATASET)

    def load_normal_processed_dataset_artifact(self):
        """This function is used to load an artifact associated with the normal processed dataset. It takes in a self argument and returns a directory artifact associated with the normal processed tag and the job of loading the dataset."""
        return self.load_dir_artifact(WB_ARTIFACT_NORMAL_PROCESSED_TAG, WB_JOB_LOAD_DATASET)

    def load_model_artifact(self, run):
        """This function loads a model from a run in W&B. 
        It takes in a parameter 'run' which is the run from which the model should be loaded. 
        The function first uses the W&B artifact associated with the run to download the model. It then returns the directory of the model."""
        artifact_wdb_path = self.__get_wb_artifact_path(WB_ARTIFACT_MODEL_TAG)
        dataset_artifact = run.use_artifact(artifact_wdb_path, type=WB_ARTIFACT_MODEL_TAG)
        model_dir = dataset_artifact.download()

        return model_dir

    def generate_model_artifact(self):
        """This code defines a function called generate_model_artifact that takes in an object of the class it is defined in as a parameter. The function creates an artifact object from the Wandb library, with the tag given by the self.artifact_model_tag parameter and type set to "model". It then adds a file located at model_path to the artifact and returns it."""
        model = wandb.Artifact(WB_ARTIFACT_MODEL_TAG, type=WB_ARTIFACT_MODEL_TAG)
        model.add_file(model_path())
        return model

    def upload_model_artifact(self, run):
        """This function uploads a model from a run in W&B. 
        It takes in a parameter 'run' which is the run from which the model should be uploaded. 
        The function first uses the W&B artifact associated with the run to upload the model. It then returns the directory of the model."""
        model_artifact = self.generate_model_artifact()
        run.log_artifact(model_artifact, aliases=[self.wdb_alias])
