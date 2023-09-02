import tensorflow as tf
from classifier import *
from hypermodel import CustomHyperModel
from dataset_representation import *
from preprocessing import *
from wandb_utils import WandbUtils
from kerastuner.oracles import BayesianOptimizationOracle, GridSearchOracle


class Main:
    def __init__(self, wdb_tags: list(), is_categorical) -> None:
        # Check the availability of gpu
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError("No GPUs available")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Initialize the WandbUtils class and start a new run
        dataset_alias = "prebuilt_multiclass_features" if is_categorical else "prebuilt_binary_features"
        self.wdb = WandbUtils(wdb_tags, dataset_alias)
        self.is_categorical = is_categorical

    def preprocessing(self, input_size, target_size, skip_to_step=None):
        # Initialize the Preprocessing class
        pp = Preprocessing(img_input_size=input_size, img_target_size=target_size)
        covid_artifact = None
        normal_artifact = None
        covid_masks_artifact = None
        normal_masks_artifact = None

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
            if not covid_artifact or not normal_artifact:
                covid_artifact, normal_artifact = load_dataset_artifacts()

            # Load the masks artifacts
            covid_mask_artifact = self.wdb.load_dataset_artifact(CovidMaskDataset())
            normal_mask_artifact = self.wdb.load_dataset_artifact(NormalMaskDataset())

            # Process the images
            pp.process_images(covid_artifact, covid_mask_artifact, normal_artifact, normal_mask_artifact)

            # Upload the processed datasets to wandb
            self.wdb.upload_dataset_artifact(CovidProcessedDataset())
            self.wdb.upload_dataset_artifact(NormalProcessedDataset())

        def extract_characteristics():
            nonlocal covid_masks_artifact, normal_masks_artifact
            # Load the processed datasets as artifacts
            covid_processed_artifact = self.wdb.load_dataset_artifact(CovidProcessedDataset())
            normal_processed_artifact = self.wdb.load_dataset_artifact(NormalProcessedDataset())

            # Load the masks artifacts for radiomics
            if not covid_masks_artifact:
                covid_masks_artifact = self.wdb.load_dataset_artifact(CovidMaskDataset())
            if not normal_masks_artifact:
                normal_masks_artifact = self.wdb.load_dataset_artifact(NormalMaskDataset())

            self.wdb.log_histogram_chart_comparison(target_size)

            # Generate characteristics file
            pp.generate_characteristics(
                covid_processed_artifact, normal_processed_artifact,
                covid_masks_artifact, normal_masks_artifact)

            # Upload characteristics
            self.wdb.upload_characteristics()

        steps = [upload_base_dataset, generate_mask_dataset, processing, extract_characteristics]

        if skip_to_step is not None:
            steps = steps[skip_to_step-1:]

        print(f"Given step:{skip_to_step}, running {steps}")

        for step in steps:
            step()
        return self

    def tuning(self):
        characteristics_artifact = self.wdb.load_characteristics()
        classifier = Classifier(characteristics_artifact=characteristics_artifact)

        if self.is_categorical:
            accuracy = tf.keras.metrics.CategoricalAccuracy()
            objective = 'val_categorical_accuracy'
            output_activation = ['softmax']
            loss = ['categorical_crossentropy']
        else:
            accuracy = tf.keras.metrics.BinaryAccuracy()
            objective = 'val_binary_accuracy'
            output_activation = ['sigmoid']
            loss = ['binary_crossentropy']

        metrics = [accuracy,
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall(),
                   tf.keras.metrics.AUC(),
                   Classifier.f1_score,
                   Classifier.custom_sensitivity,
                   Classifier.custom_specificity]

        # hypermodel = CustomHyperModel(
        #     metrics=metrics,
        #     optimizer_callout=lambda hp: hp.Choice("optimizer", values=["adam", "sgd", "rmsprop"]),
        #     activation_callout=lambda hp: hp.Choice("activation", values=['relu', 'elu', 'selu']),
        #     activation_output_callout=lambda hp: hp.Choice("activation_output", output_activation),
        #     loss_callout=lambda hp: hp.Choice("loss", values=loss),
        #     dropout_callout=lambda hp: hp.Float("dropout", min_value=0.1, max_value=0.3, step=0.05),
        #     learning_rate_callout=lambda hp: hp.Float("learning_rate", min_value=1e-6, max_value=1e-2, step=1e-4),
        #     dense_layers_callout=lambda hp: hp.Int("num_layers", min_value=4, max_value=20, step=1),
        #     filters_callout=lambda hp: hp.Int("filters", min_value=8, max_value=64, step=8),
        #     kernel_size_callout=lambda hp: hp.Int("kernel_size", min_value=3, max_value=5, step=1),
        #     pool_size_callout=lambda hp: hp.Int("pool_size", min_value=2, max_value=4, step=1),
        #     conv_layers_callout=lambda hp: hp.Int("conv_layers", min_value=1, max_value=4, step=1),
        #     units_callout=lambda hp: hp.Int("units", min_value=32, max_value=500, step=16),
        #     use_same_units_callout=lambda hp: hp.Boolean("use_same_units")
        # )

        oracle = BayesianOptimizationOracle(
            objective=objective,
            max_trials=700
        )

        # Define the list of argument values
        arg_values = [
            ["rmsprop", "elu", output_activation[0], loss[0], 0.1, 0.008201, 4, 64, 4, 2, 2, 32, False],
            ["rmsprop", "selu", "softmax", "categorical_crossentropy", 0.1, 0.004801, 8, 48, 4, 2, 2, 208, True],
            ["adam", "relu", "softmax", "categorical_crossentropy", 0.15, 0.006201, 9, 64, 4, 4, 1, 32, False],
            ["adam", "relu", "softmax", "categorical_crossentropy", 0.2, 0.001401, 11, 32, 3, 2, 3, 352, False],
            ["adam", "relu", "softmax", "categorical_crossentropy", 0.1, 0.009801, 9, 8, 3, 3, 1, 496, False]
        ]

        # Define the argument names
        arg_names = [
            "optimizer_callout", "activation_callout", "activation_output_callout", "loss_callout", "dropout_callout", "learning_rate_callout",
            "dense_layers_callout", "filters_callout", "kernel_size_callout", "pool_size_callout", "conv_layers_callout", "units_callout", "use_same_units_callout"
        ]

        # Create a list of dictionaries with keyword arguments for each model
        params = [dict(zip(arg_names, values)) for values in arg_values]

        def batch_size_callout(hp): return hp.Int("batch_size", min_value=1024, max_value=1024, step=4)
        # classifier.tune(hypermodel, oracle, 3000, objective, batch_size_callout, self.wdb)
        classifier.cross_validation(batch_size=1024,
                                    epochs=200,
                                    wdb=self.wdb,
                                    metrics=metrics,
                                    params=params)
        return self

    def finish(self): self.wdb.finish()


# RUN
main = Main(["cross_val_test"], is_categorical=True)
try:
    # main.preprocessing(input_size=(512, 512, 1), target_size=(512, 512), skip_to_step=4)
    main.tuning()
finally:
    main.finish()

print("Finish")
