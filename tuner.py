import kerastuner as kt
import wandb
from wandb.keras import WandbCallback


class CustomTuner(kt.Tuner):
    """
    Custom Tuner subclassed from `kt.Tuner`
    """

    def __init__(self, batch_size_callout, *args, **kwargs):
        self.batch_size_callout = batch_size_callout
        super().__init__(*args, **kwargs)

    def run_trial(self, trial, trainX, trainY, epochs, objective):
        hp = trial.hyperparameters
        objective_name_str = objective

        # create the model with the current trial hyperparameters
        model = self.hypermodel.build(hp)

        # Combine the current trial hyperparameters with the hyperparameters used to build the hypermodel
        batch_size = self.batch_size_callout(hp)

        # Initiates new run for each trial on the dashboard of Weights & Biases
        run = wandb.init(project="tcc_code", config={**hp.values}, group="trial")

        # WandbCallback() logs all the metric data such as
        # loss, accuracy and etc on dashboard for visualization
        history = model.fit(trainX,
                            trainY,
                            batch_size=batch_size[0],
                            epochs=epochs,
                            validation_split=0.1,
                            workers=6,
                            use_multiprocessing=True,
                            callbacks=[WandbCallback()])
        # if val_accurcy used, use the val_accuracy of last epoch model which is fully trained
        val_acc = history.history['val_accuracy'][-1]  # [-1] will give the last value in the list

        # Send the objective data to the oracle for comparison of hyperparameters
        self.oracle.update_trial(trial.trial_id, {objective_name_str: val_acc})

        # ends the run on the Weights & Biases dashboard
        run.finish()
