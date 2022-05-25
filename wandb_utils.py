from glob import glob
import numpy as np
import wandb
from image import Image, ImageGenerator
from utils import *


class WandbUtils:
    @staticmethod
    def log_raw_images_artifacts():
        with(wandb.init(project='tcc', job_type="upload-artifacts")) as run:
            raw_data = wandb.Artifact("raw-files", type="dataset")
            raw_data.add_dir(covid_path(), "covid")
            raw_data.add_dir(non_covid_path(), "non-covid")
            run.log_artifact(raw_data)

        print("Artifact raw-files added")

    @staticmethod
    def generate_model_artifact():
        model = wandb.Artifact("model", type="model")
        model.add_file(model_path, "model.h5")
        return model

    @staticmethod
    def log_interactive_table():
        with(wandb.init(project='tcc', job_type="upload-artifacts")) as run:
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

            create_wandb_table(covid_images(), covid_masks(),
                               cov_processed(), "covid")
            create_wandb_table(
                non_covid_images(), non_covid_masks(), non_cov_processed(), "non-covid")

            run.finish()

    @staticmethod
    def download_artifact(name, relative_path):
        with(wandb.init(project='tcc', job_type="download-artifacts")) as run:
            artifact = run.use_artifact('vhviveiros/tcc/' + name)
            artifact.download(root=abs_path(relative_path))
            print("Artifact " + name + " downloaded")

    @staticmethod
    def log_histogram_chart_comparison():
        with(wandb.init(project='tcc', job_type="upload-artifacts")) as run:
            cov_processed_gen = ImageGenerator().generate_from(cov_processed_path())
            non_cov_processed_gen = ImageGenerator().generate_from(non_cov_processed_path())

            cov_hist_data = np.transpose(
                [x.hist() for x in cov_processed_gen]).tolist()
            non_cov_hist_data = np.transpose(
                [x.hist() for x in non_cov_processed_gen]).tolist()

            cov_hist_data = list(zip([*range(1, 256)], cov_hist_data))
            non_cov_hist_data = list(zip([*range(1, 256)], non_cov_hist_data))

            cov_hist_data = [list(x) for x in list(cov_hist_data)]
            non_cov_hist_data = [list(x) for x in list(non_cov_hist_data)]

            cov_table = wandb.Table(data=cov_hist_data, columns=[
                "Intensity", "Value"])
            non_cov_table = wandb.Table(
                data=non_cov_hist_data, columns=["Intensity", "Value"])

            # fields = {"x": "Intensity", "value": "Value"}

            # cov_chart = wandb.plot_table(vega_spec_name="tcc/cov_chart",
            #                              data_table=cov_table,
            #                              fields=fields)
            # non_cov_chart = wandb.plot_table(vega_spec_name="tcc/non_cov_chart",
            #                                  data_table=non_cov_table,
            #                                  fields=fields)

            run.log({"cov_chart": cov_table})
            run.log({"non_cov_chart": non_cov_table})
