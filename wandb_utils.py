from glob import glob
import numpy as np
import wandb
from image import Image
from utils import abs_path


class ArtifactMng:
    @staticmethod
    def log_raw_images_data():
        covid_path = abs_path('dataset/covid')
        non_covid_path = abs_path('dataset/normal')

        with(wandb.init(project='tcc', job_type="upload-artifacts")) as run:
            raw_data = wandb.Artifact("raw-files", type="dataset")
            raw_data.add_dir(covid_path, "covid")
            raw_data.add_dir(non_covid_path, "non-covid")
            run.log_artifact(raw_data)

        print("Artifact raw-files added")

    @staticmethod
    def log_interactive_images_data():
        covid_path = abs_path('dataset/covid')
        covid_masks_path = abs_path('cov_masks')

        non_covid_path = abs_path('dataset/normal')
        non_covid_masks_path = abs_path('non_cov_masks')

        cov_processed_path = abs_path('cov_processed')
        non_cov_processed_path = abs_path('non_cov_processed')

        with(wandb.init(project='tcc', job_type="upload-artifacts")) as run:
            covid_images = glob(covid_path + "/*g")
            non_covid_images = glob(non_covid_path + "/*g")
            covid_masks = glob(covid_masks_path + "/*g")
            non_covid_masks = glob(non_covid_masks_path + "/*g")
            cov_processed = glob(cov_processed_path + "/*g")
            non_cov_processed = glob(non_cov_processed_path + "/*g")

            if (len(covid_images) == len(covid_masks) == len(cov_processed)):
                data = np.asarray([covid_images, covid_masks])
                for i in data.T:
                    img = wandb.Image(Image(i[0]).data, masks={
                        "mask": {
                            "mask_data": np.asarray(Image(i[1]).data),
                            "class_labels": {
                                1: "Mask",
                            }
                        }})
                    run.log({"Image": img})

            if (len(non_covid_images) == len(non_covid_masks) == len(non_cov_processed)):
                data = np.asarray([non_covid_images, non_covid_masks])
                for i in data.T:
                    img = wandb.Image(Image(i[0]).data, masks={
                        "mask": {
                            "mask_data": np.asarray(Image(i[1]).data),
                            "class_labels": {
                                1: "Mask",
                            }
                        }})
                    run.log({"Image": img})

    @staticmethod
    def download_artifact(name, relative_path):
        with(wandb.init(project='tcc', job_type="download-artifacts")) as run:
            artifact = run.use_artifact('vhviveiros/tcc/' + name)
            artifact.download(root=abs_path(relative_path))
            print("Artifact " + name + " downloaded")
