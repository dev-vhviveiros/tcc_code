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
        with (wandb.init(project=self.project_name, job_type=job_type)) as run:
            callback(run)

        print("Job: <" + job_type + "> done!")

    def __create_wandb_table(self, images, masks, processed, tag):
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
        def callback(run):
            self.__create_wandb_table(cov_images(), cov_masks(), cov_processed(), "covid")
            self.__create_wandb_table(non_cov_images(), non_cov_masks(), non_cov_processed(), "non-covid")
            run.finish()

        self.execute_with(callback, self.job_log_table)

    def log_histogram_chart_comparison(self):
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
        model = wandb.Artifact(self.artifact_model_tag, type="model")
        model.add_file(model_path, "model.h5")
        return model

    def upload_dataset_artifact(self, alias=""):
        def callback(run):
            raw_data = wandb.Artifact(self.artifact_dataset_tag, type="dataset")
            raw_data.add_dir(dataset_path(), "dataset")
            run.log_artifact(raw_data, aliases=[alias])

        self.execute_with(callback, self.job_upload_dataset)

    def download_artifact(self, name, relative_path, alias='latest'):
        api = wandb.Api()
        artifact = api.artifact('vhviveiros/' + self.project_name + '/' + name + ":" + alias)
        artifact.download(root=abs_path(relative_path))
        print("Artifact " + name + " downloaded")

    def load_artifacts(self, run):
        dataset_artifact = run.use_artifact('%s/%s:%s'
                                            % (self.project_path, self.artifact_dataset_tag, self.wdb_data_alias),
                                            type='dataset')
        dataset_dir = dataset_artifact.download()

        return [dataset_dir]
