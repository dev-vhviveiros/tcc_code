import kerastuner as kt
import wandb
from wandb.keras import WandbCallback

from dataset_representation import COVID_TAG, DATASET_TAG


class CustomTuner(kt.Tuner):
    """
    A custom tuner subclassed from `kt.Tuner` that uses Weights & Biases for logging and visualization.
    """

    def __init__(self, batch_size_callout, *args, **kwargs):
        """
        Initializes a new instance of the CustomTuner class.

        Args:
            batch_size_callout: A callable that takes a `HyperParameters` object and returns a batch size.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.batch_size_callout = batch_size_callout
        super().__init__(*args, **kwargs)

    def run_trial(self, trial, trainX, trainY, epochs, objective):
        """
        Runs a single trial with the given hyperparameters.

        Args:
            trial: A `Trial` object representing the current trial.
            trainX: The training data.
            trainY: The training labels.
            epochs: The number of epochs to train for.
            objective: The name of the objective metric.

        Returns:
            None.
        """
        # Get the hyperparameters for the current trial
        hp = trial.hyperparameters

        # Get the name of the objective metric
        objective_name_str = objective

        # Create the model with the current trial hyperparameters
        model = self.hypermodel.build(hp)

        # Get the batch size for the current trial hyperparameters
        batch_size = self.batch_size_callout(hp)

        # Initiates new run for each trial on the dashboard of Weights & Biases
        with wandb.init(project="tcc_code", config={**hp.values}, group="trial") as run:
            # Use WandbCallback() to log all the metric data such as loss, accuracy, etc. on the Weights & Biases dashboard for visualization
            history = model.fit(trainX,
                                trainY,
                                batch_size=batch_size[0],
                                epochs=epochs,
                                validation_split=0.1,
                                workers=6,
                                use_multiprocessing=True,
                                callbacks=[WandbCallback()])

            # Get the validation objective of the last epoch model which is fully trained
            val_acc = history.history[f'val_{objective}'][-1]

            # Send the objective data to the oracle for comparison of hyperparameters
            self.oracle.update_trial(trial.trial_id, {objective_name_str: val_acc})

            # End the run on the Weights & Biases dashboard
            run.finish()
