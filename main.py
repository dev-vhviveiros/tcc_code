import tensorflow as tf
from classifier import *
from hypermodel import CustomHyperModel
from dataset_representation import *
from preprocessing import *
from wandb_utils import WandbUtils
from kerastuner.oracles import BayesianOptimizationOracle


class Main:
    def __init__(self) -> None:
        # Check the availability of gpu
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError("No GPUs available")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Initialize the WandbUtils class and start a new run
        self.wdb = WandbUtils("basic")

    def preprocessing(self, input_size=(256, 256, 1), target_size=(256, 256), skip_to_step=None) -> 'Main':
        # Initialize the Preprocessing class
        pp = Preprocessing(img_input_size=input_size, img_target_size=target_size)
        covid_artifact = None
        normal_artifact = None

        def upload_base_dataset():
            # Upload the dataset artifacts
            self.wdb.upload_dataset_artifact(CovidDataset())
            self.wdb.upload_dataset_artifact(NormalDataset())

        def load_dataset_artifacts():
            # Load the dataset artifacts
            covid_artifact = self.wdb.load_dataset_artifact(CovidDataset())
            normal_artifact = self.wdb.load_dataset_artifact(NormalDataset())
            return (covid_artifact, normal_artifact)

        def generate_mask_dataset():
            # Load the dataset artifacts
            covid_artifact, normal_artifact = load_dataset_artifacts()

            # Generate lung masks for all images
            pp.generate_lungs_masks(covid_artifact, normal_artifact)

            # Upload the masks artifacts
            self.wdb.upload_dataset_artifact(CovidMaskDataset())
            self.wdb.upload_dataset_artifact(NormalMaskDataset())

        def processing():
            # Load the dataset artifacts if they are not already loaded
            nonlocal covid_artifact, normal_artifact
            if covid_artifact is None or normal_artifact is None:
                covid_artifact, normal_artifact = load_dataset_artifacts()

            # Load the masks artifacts
            covid_mask_artifact = self.wdb.load_dataset_artifact(CovidMaskDataset())
            normal_mask_artifact = self.wdb.load_dataset_artifact(NormalMaskDataset())

            # Process the images
            pp.process_images(covid_artifact, covid_mask_artifact, normal_artifact, normal_mask_artifact)

            # Upload the processed datasets to wandb
            self.wdb.upload_dataset_artifact(CovidProcessedDataset())
            self.wdb.upload_dataset_artifact(NormalProcessedDataset())

        def generate_hist_and_characteristics():
            # Load the processed datasets as artifacts
            covid_processed_artifact = self.wdb.load_dataset_artifact(CovidProcessedDataset())
            normal_processed_artifact = self.wdb.load_dataset_artifact(NormalProcessedDataset())

            self.wdb.log_histogram_chart_comparison(target_size)

            # Generate characteristics file
            pp.generate_characteristics(covid_processed_artifact, normal_processed_artifact)

            # Upload characteristics
            self.wdb.upload_characteristics()

        steps = [upload_base_dataset, generate_mask_dataset, processing, generate_hist_and_characteristics]

        if skip_to_step is not None:
            steps = steps[skip_to_step-1:]

        print(f"Given step:{skip_to_step}, running {steps}")

        for step in steps:
            step()
        return self

    def tuning(self, num_samples) -> 'Main':
        characteristics_artifact = self.wdb.load_characteristics()
        classifier = Classifier(characteristics_artifact=characteristics_artifact, num_samples=num_samples)

        metrics = ['accuracy',
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall(),
                   tf.keras.metrics.AUC(),
                   Classifier.custom_sensitivity,
                   Classifier.custom_specificity]

        hypermodel = CustomHyperModel(
            metrics=metrics,
            optimizer_callout=lambda hp: hp.Choice("optimizer", values=["sgd", "adam", "adadelta"]),
            activation_callout=lambda hp: hp.Choice(
                "activation", values=['relu', 'elu', 'selu', 'tanh', 'softsign', 'softplus']),
            activation_output_callout=lambda hp: hp.Choice(
                "activation_output", values=['sigmoid', 'softmax', 'tanh', 'softplus']),
            loss_callout=lambda hp: hp.Choice(
                "loss", values=['mean_squared_error', 'kl_divergence', 'poisson', 'binary_crossentropy']),
            dropout_callout=lambda hp: hp.Float("dropout", min_value=0.15, max_value=0.3, step=0.05),
            units_callout=lambda hp: hp.Int("units", min_value=50, max_value=500, step=50),
            learning_rate_callout=lambda hp: hp.Float("learning_rate", min_value=1e-6, max_value=1e-2, step=1e-4)
        )

        objective = 'val_accuracy'

        oracle = BayesianOptimizationOracle(
            objective=objective,
            max_trials=500
        )

        def batch_size_callout(hp): return hp.Int("batch_size", min_value=8, max_value=256, step=4)
        classifier.tune(hypermodel, oracle, 1000, objective, batch_size_callout)
        return self

    def finish_wdb(self): self.wdb.finish()


# RUN
try:
    main = Main()
    main.preprocessing(input_size=(256, 256, 1), target_size=(256, 256), skip_to_step=3)
    main.tuning(3616)
finally:
    main.finish_wdb()
