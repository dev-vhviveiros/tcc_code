import numpy as np
import wandb
import os
from image import Image, ImageGenerator
from utils import abs_path, dataset_path, model_path, cov_processed_path, non_cov_processed_path
from utils import cov_processed, cov_images, cov_masks
from utils import non_cov_processed, non_cov_images, non_cov_masks


class WandbUtils:
    job_upload_dataset = "upload_dataset"
    job_log_table = "log_interactive_table"
    job_histogram_chart = "log_histogram_chart"
    job_load_artifacts = "load_artifacts"
    artifact_dataset_tag = "dataset"
    artifact_model_tag = "model"

    def __init__(self, wdb_data_alias, project_owner="vhviveiros", project_name="tcc"):
        self.wdb_data_alias = wdb_data_alias
        self.project_path = project_owner + "/" + project_name
        self.project_name = project_name

    def execute_with(self, callback, job_type):
        """This code takes two parameters, self and callback. It then initializes a run with the project name and job type specified in the parameters. The callback function is then called with the run as an argument. Finally, it prints a message indicating that the job is done."""
        with (wandb.init(project=self.project_name, job_type=job_type)) as run:
            callback(run)

        print("Job: <" + job_type + "> done!")

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

    def log_interactive_table(self):
        """ The function takes in the self parameter, which is a reference to the current instance of the class. Inside the function, a callback function is defined that takes in a run parameter. This callback function calls two other functions, __create_wandb_table and finish(), which create an interactive table in W&B with covid and non-covid images and masks, respectively. Finally, the execute_with() method is called with the callback and job_log_table parameters."""
        def callback(run):
            self.__create_wandb_table(cov_images(), cov_masks(), cov_processed(), "covid")
            self.__create_wandb_table(non_cov_images(), non_cov_masks(), non_cov_processed(), "non-covid")
            run.finish()

        self.execute_with(callback, self.job_log_table)

    def log_histogram_chart_comparison(self):
        """This function creates a log of histogram chart comparison. It first creates two image generators from cov_processed_path and non_cov_processed_path. Then it creates two histogram data sets from the generated images, cov_hist_data and non_cov_hist_data. The data is then converted to a list of tuples and stored in cov_table and non_cov_table. Finally, the tables are logged in the run."""
        def callback(run):
            cov_processed_gen = ImageGenerator().generate_from(cov_processed_path())
            non_cov_processed_gen = ImageGenerator().generate_from(non_cov_processed_path())

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
        self.execute_with(callback, self.job_histogram_chart)

    def generate_model_artifact(self):
        """This code defines a function called generate_model_artifact that takes in an object of the class it is defined in as a parameter. The function creates an artifact object from the Wandb library, with the tag given by the self.artifact_model_tag parameter and type set to "model". It then adds a file located at model_path to the artifact and returns it."""
        model = wandb.Artifact(self.artifact_model_tag, type="model")
        model.add_file(model_path, "model.h5")
        return model

    def upload_dataset_artifact(self, alias=""):
        """This function uploads a dataset artifact to W&B. It takes in an optional alias parameter. The callback function creates a wandb.Artifact object with the tag "artifact_dataset_tag" and type "dataset". It then adds the directory at the path "dataset_path" to the artifact and logs it with the given alias. Finally, it executes the job "job_upload_dataset" with the callback function."""
        def callback(run):
            raw_data = wandb.Artifact(self.artifact_dataset_tag, type="dataset")
            raw_data.add_dir(dataset_path(), "dataset")
            run.log_artifact(raw_data, aliases=[alias])

        self.execute_with(callback, self.job_upload_dataset)

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

    def load_dataset(self, run):
        """This code is used to load a dataset from a project path. It takes in the parameter "run" and creates an artifact_wdb_path string using the project path, artifact dataset tag, and wdb data alias. It then uses the run parameter to use the artifact_wdb_path as an artifact with a type of the artifact dataset tag. Finally, it downloads the dataset from the artifact and returns it as a list."""
        artifact_wdb_path = '%s/%s:%s' % (self.project_path, self.artifact_dataset_tag, self.wdb_data_alias)
        dataset_artifact = run.use_artifact(artifact_wdb_path, type=self.artifact_dataset_tag)
        dataset_dir = dataset_artifact.download()

        return [dataset_dir]

    def load_model(self, run):
        """This function is used to load a model from a project path, tag, and alias. It takes in the argument 'run', which is an object containing information about the run. It then creates a path to the artifact model using the project path, tag, and alias. It then creates a dataset artifact using this path and type of artifact model tag. Finally, it downloads the model from this dataset artifact and returns it."""
        artifact_wdb_path = '%s/%s:%s' % (self.project_path, self.artifact_model_tag, self.wdb_data_alias)
        dataset_artifact = run.use_artifact(artifact_wdb_path, type=self.artifact_model_tag)
        model_dir = dataset_artifact.download()

        return model_dir

    def upload_model(self, run):
        #TODO
        return
