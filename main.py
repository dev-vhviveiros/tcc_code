import tensorflow as tf
from classifier import *
from hypermodel import CustomHyperModel
from dataset_representation import *
from preprocessing import *
from wandb_utils import WandbUtils
from kerastuner.oracles import BayesianOptimizationOracle, RandomSearchOracle
from tuner import CustomTuner

# Check the availability of gpu
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # Initialize the WandbUtils class and start a new run
    wdb = WandbUtils("basic")
    try:
        # # Load the dataset artifacts
        # covid_artifact = wdb.load_dataset_artifact(CovidDataset())
        # normal_artifact = wdb.load_dataset_artifact(NormalDataset())

        # # Initialize the Preprocessing class and generate lung masks for all images
        # pp = Preprocessing(wdb)
        # pp.generate_lungs_masks(covid_artifact, normal_artifact)

        # # Load the masks artifacts
        # covid_mask_artifact = wdb.load_dataset_artifact(CovidMaskDataset())
        # normal_mask_artifact = wdb.load_dataset_artifact(NormalMaskDataset())

        # # Process the images
        # pp.process_images(covid_artifact, covid_mask_artifact, normal_artifact, normal_mask_artifact)

        # # Load the processed datasets as artifacts
        # covid_processed_artifact = wdb.load_dataset_artifact(CovidProcessedDataset())
        # normal_processed_artifact = wdb.load_dataset_artifact(NormalProcessedDataset())

        # wdb.log_histogram_chart_comparison()

        # # Generate characteristics file
        # pp.generate_characteristics(covid_processed_artifact, normal_processed_artifact)
        characteristics_artifact = wdb.load_characteristics()
        classifier = Classifier(characteristics_artifact=characteristics_artifact)

        metrics = ['accuracy',
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall(),
                   tf.keras.metrics.AUC(),
                   Classifier.custom_sensitivity,
                   Classifier.custom_specificity]

        hypermodel = CustomHyperModel(
            metrics=metrics,
            optimizer_callout=lambda hp: hp.Choice("optimizer", values=["sgd"]),
            activation_callout=lambda hp: hp.Choice("activation", values=["relu"]),
            activation_output_callout=lambda hp: hp.Choice("activation_output", values=["sigmoid"]),
            loss_callout=lambda hp: hp.Choice("loss", values=["binary_crossentropy"]),
            dropout_callout=lambda hp: hp.Float("dropout", min_value=0.2, max_value=0.2, step=0.05),
            units_callout=lambda hp: hp.Int("units", min_value=180, max_value=180, step=50)
        )

        # tuner = BayesianOptimization(
        #     hypermodel,
        #     objective='val_accuracy',
        #     executions_per_trial=2,
        #     directory='tuner',
        #     project_name='tcc',
        #     overwrite=True
        # )

        objective = 'val_accuracy'

        oracle = RandomSearchOracle(
            objective=objective,
            max_trials=2,
        )

        # TODO next steps:
        # Integrate keras tuner to the wandb
        def batch_size_callout(hp): return hp.Int("batch_size", min_value=8, max_value=16, step=4),

        wdb.finish()
        classifier.tune(hypermodel, oracle, 130, objective, batch_size_callout)
        # classifier.fit(logs_folder='./logs/', export_model=False)
    finally:
        pass
else:
    print("No GPUs available")
