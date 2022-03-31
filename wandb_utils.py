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
    def log_interactive_table():
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

            def create_wandb_table(images, masks, processed, type):
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
                wandb.log({type + " Table": table})

            create_wandb_table(covid_images, covid_masks,
                               cov_processed, "covid")
            create_wandb_table(
                non_covid_images, non_covid_masks, non_cov_processed, "non-covid")

            run.finish()

    @staticmethod
    def download_artifact(name, relative_path):
        with(wandb.init(project='tcc', job_type="download-artifacts")) as run:
            artifact = run.use_artifact('vhviveiros/tcc/' + name)
            artifact.download(root=abs_path(relative_path))
            print("Artifact " + name + " downloaded")
