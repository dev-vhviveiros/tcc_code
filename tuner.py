import kerastuner as kt
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from hypermodel import HyperModel
from wandb_utils import WandbUtils
from tensorflow.keras.utils import to_categorical


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

    def run_trial(self, trial, x, y, epochs, objective, validation_split, wandb_utils: WandbUtils):
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

        try:
            if objective == 'val_categorical_accuracy':
                y = to_categorical(y, 3)

            # Create the model with the current trial hyperparameters
            model: HyperModel = self.hypermodel.build(hp)

            # Get the batch size for the current trial hyperparameters
            batch_size = self.batch_size_callout(hp)

            # Print the number of training and validation samples
            num_samples = len(x)
            print(f"Trial {trial.trial_id}: Training on {num_samples * (1 - validation_split)} samples, validating on ~{validation_split * num_samples} samples, with {validation_split} validation_split")

            # Print the sum of training and validation samples
            print(f"Trial {trial.trial_id}: Total samples: {num_samples}")

            # Reshape the input data to add a new axis
            x_reshaped = x[..., np.newaxis]

            # Initiates new run for each trial on the dashboard of Weights & Biases
            with wandb.init(project="tcc_code", config={**hp.values}, group="trial", tags=wandb_utils.wdb_tags) as run:
                # Use WandbCallback() to log all the metric data such as loss, accuracy, etc. on the Weights & Biases dashboard for visualization
                early_stop = EarlyStopping(monitor=objective, patience=200, restore_best_weights=True, mode="max")
                history = model.fit(x_reshaped, y,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_split=validation_split,
                                    workers=6,
                                    use_multiprocessing=True,
                                    callbacks=[early_stop, WandbCallback(
                                        save_model=False,
                                        monitor=objective,
                                        mode='max'
                                    )])

                def get_metric_value(metric_name, direction):
                    values = history.history[metric_name]
                    return max(values) if direction == 'max' else min(values)

                # Define metrics along with their optimization directions
                metrics = [
                    (objective, 'max'),
                    ('val_precision', 'max'),
                    ('val_recall', 'max'),
                    ('val_auc', 'max'),
                    ('val_custom_sensitivity', 'max'),
                    ('val_custom_specificity', 'max'),
                    ('val_loss', 'min')
                ]

                # Create a dictionary to hold the metric values for the best epoch
                metric_values = {metric_name: get_metric_value(metric_name, direction)
                                 for metric_name, direction in metrics}

                # Send the objective data to the oracle for comparison of hyperparameters
                self.oracle.update_trial(trial.trial_id, metric_values)

        except Exception as e:
            print(e)
            self.oracle.update_trial(trial.trial_id, {objective: 0})
            return

        finally:
            # End the run on the Weights & Biases dashboard
            run.finish()
