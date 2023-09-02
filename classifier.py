import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, KFold
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from hypermodel import CustomHyperModel
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from wandb.keras import WandbCallback
import wandb

from tuner import CustomTuner
from utils import load_config
from wandb_utils import WandbUtils


class Classifier:
    """
    Use: 
    input_file = your_file.csv when you want to use a new model
    import_model = if you saved a model and want to import it
    """

    def __init__(self, characteristics_artifact: str = None, model_artifact: str = None):
        """
        Initializes the Classifier object.

        Args:
            characteristics_artifact (str, optional): The characteristics artifact. Defaults to None.
            model_artifact (str, optional): The model artifact. Defaults to None.
        """
        # Load the WandB project name from the configuration file
        self.wdb_project = load_config("wb_project_name")

        num_samples = -1

        # Load the characteristics artifact if provided
        if characteristics_artifact is not None:
            num_samples = self.__load_characteristics(characteristics_artifact)

        # Load the model artifact if provided
        if model_artifact is not None:
            self.__import_model(model_artifact)

        # Print a message indicating the project and artifacts being used
        print(f"Initializing classifier project: {self.wdb_project} with:\n"
              f"Characteristics: {characteristics_artifact}\n"
              f"Model: {model_artifact}\n",
              f"Number of samples per label: {num_samples}")

    def __load_characteristics(self, characteristics_artifact):
        """
        Loads the image characteristics and labels from a CSV file, splits the data into training and testing sets, and normalizes the image characteristics using the StandardScaler function.

        Args:
            characteristics_artifact (str): The path to the CSV file containing the image characteristics and labels.
            num_samples (int): The number of samples to load for each label.
            test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.2.
        Returns:
            None
        """
        # Read the CSV file containing the image characteristics
        characteristics_df = pd.read_csv(characteristics_artifact)
        print("SHAPE:" + str(characteristics_df.shape))

        # Group the DataFrame by the label column
        grouped = characteristics_df.groupby(characteristics_df.columns[-1])

        # Define the number of classes
        self.num_classes = len(grouped)

        # Define the num_samples based on the label with less samples
        num_samples = min([len(group) for _, group in grouped])

        # Create a new DataFrame with the first <num_samples> elements from each group
        characteristics_df = pd.concat([group.iloc[:num_samples] for _, group in grouped])

        # Extract the input data (image characteristics) and output data (labels)
        image_characteristics = characteristics_df.iloc[:, :-1].values
        self.labels = characteristics_df.iloc[:, -1].values

        # Normalize the data using the MinMaxScaler function
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_characteristics = scaler.fit_transform(image_characteristics)

        # Select KBest features
        kbest = SelectKBest(chi2, k=100)
        kbest.fit(normalized_characteristics, self.labels)
        self.features = kbest.transform(normalized_characteristics)

        return num_samples

    def categorize_labels(self, labels):
        """
        Convert the labels of the dataset to categorical format if necessary.

        If the number of classes in the dataset is greater than 2, the labels are converted to categorical using the
        `to_categorical` function from Keras.

        Args:
            labels: The labels of the dataset.

        Returns:
            The labels of the dataset in categorical format, if necessary.
        """
        # Convert the labels to categorical, if num_classes > 2
        if self.num_classes > 2:
            labels = to_categorical(labels, num_classes=self.num_classes)
        return labels

    @staticmethod
    def custom_specificity(y_true, y_pred):
        """
        Calculates the specificity of a binary classification model.

        Args:
            y_true (tensor): The true labels.
            y_pred (tensor): The predicted labels.

        Returns:
            float: The specificity of the model.
        """
        # Calculate the true negatives and false positives using Keras backend functions
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

        # Calculate the specificity as the ratio of true negatives to the sum of true negatives and false positives
        specificity = tn / (tn + fp + K.epsilon())

        return specificity

    @staticmethod
    def custom_sensitivity(y_true, y_pred):
        """
        Calculates the sensitivity of a binary classification model.

        Args:
            y_true (tensor): The true labels.
            y_pred (tensor): The predicted labels.

        Returns:
            float: The sensitivity of the model.
        """
        # Calculate the false negatives and true positives using Keras backend functions
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        # Calculate the sensitivity as the ratio of true positives to the sum of true positives and false negatives
        sensitivity = tp / (tp + fn + K.epsilon())

        return sensitivity

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calculates the F1-score of a binary classification model.

        Args:
            y_true (tensor): The true labels.
            y_pred (tensor): The predicted labels.

        Returns:
            float: The F1-score of the model.
        """
        # Calculate the true positives, false positives, and false negatives using Keras backend functions
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

        # Calculate the precision and recall using the true positives, false positives, and false negatives
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        # Calculate the F1-score as the harmonic mean of precision and recall
        f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())

        return f1_score

    def plot_confusion_matrix(self, title: str, cmap=None, normalize: bool = False, save_dir: str = None):
        """
        Plots a confusion matrix for the model's predictions on the testing set.

        Args:
            title (str): The title of the plot.
            cmap (str, optional): The colormap to use for the plot. Defaults to None.
            normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
            save_dir (str, optional): The directory to save the plot in. Defaults to None.

        Returns:
            None
        """
        # Make predictions on the testing set and calculate the confusion matrix
        predicted_classes = self.model.predict_classes(self.normalized_testing_image_characteristics)
        confusion_matrix = confusion_matrix(predicted_classes, self.val_labels)

        # Define the target names and calculate the accuracy and misclassification rate
        target_names = ['Covid-19', 'Normal']
        accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
        misclass = 1 - accuracy

        # Set the colormap if not provided
        if cmap is None:
            cmap = plt.get_cmap('Blues')

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        # Add axis labels and tick marks if target names are provided
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        # Normalize the confusion matrix if requested
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

        # Set the threshold for text color based on normalization
        thresh = confusion_matrix.max() / 1.5 if normalize else confusion_matrix.max() / 2

        # Add text to the plot indicating the values in the confusion matrix
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(confusion_matrix[i, j]),
                         horizontalalignment="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(confusion_matrix[i, j]),
                         horizontalalignment="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black")

        # Add axis labels and a legend
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
            accuracy, misclass))
        if save_dir is not None:
            plt.savefig(save_dir)
        plt.show()

    def predict(self, x):
        """
        Predicts the classes of the given input using the model attribute of the class.

        Args:
            x (numpy.ndarray): The input data to predict the classes for.

        Returns:
            numpy.ndarray: The predicted classes.
        """
        # Use the model attribute to predict the classes of the input data
        predicted_classes = self.model.predict_classes(x)

        # Print the predicted classes and return them
        print(predicted_classes)
        return predicted_classes

    def tune(self, hypermodel, oracle, epochs: int, objective: str, batch_size_callout, wandb_utils):
        """
        Uses a custom tuner to search for the best hyperparameters for a given hypermodel.

        Args:
            hypermodel (keras_tuner.HyperModel): The hypermodel to tune.
            oracle (keras_tuner.Oracle): The oracle to use for the search.
            epochs (int): The number of epochs to train the model for during each trial.
            objective (str): The name of the metric to optimize during the search.
            batch_size_callout (Callable): A function that returns the batch size for each trial.

        Returns:
            None
        """
        # Create a custom tuner with the given hypermodel, oracle, and batch size callout function
        tuner = CustomTuner(
            batch_size_callout=batch_size_callout,
            hypermodel=hypermodel,
            executions_per_trial=2,
            directory='tuner',
            overwrite=True,
            oracle=oracle
        )

        # Search for the best hyperparameters using the custom tuner
        tuner.search(self.features, self.categorize_labels(self.labels), epochs=epochs, objective=objective,
                     validation_split=0.2, wandb_utils=wandb_utils)

    def __build_model(self, **kwargs):
        model = CustomHyperModel(
            metrics=kwargs["metrics"],
            optimizer_callout=lambda _: kwargs["optimizer_callout"],
            activation_callout=lambda _: kwargs["activation_callout"],
            activation_output_callout=lambda _: kwargs["activation_output_callout"],
            loss_callout=lambda _: kwargs["loss_callout"],
            dropout_callout=lambda _: kwargs["dropout_callout"],
            learning_rate_callout=lambda _: kwargs["learning_rate_callout"],
            dense_layers_callout=lambda _: kwargs["dense_layers_callout"],
            filters_callout=lambda _: kwargs["filters_callout"],
            kernel_size_callout=lambda _: kwargs["kernel_size_callout"],
            pool_size_callout=lambda _: kwargs["pool_size_callout"],
            conv_layers_callout=lambda _: kwargs["conv_layers_callout"],
            units_callout=lambda _: kwargs["units_callout"],
            use_same_units_callout=lambda _: kwargs["use_same_units_callout"]
        )

        return model.build(None)

    def cross_validation(self, **kwargs):
        batch_size = kwargs["batch_size"]
        epochs = kwargs["epochs"]
        wdb: WandbUtils = kwargs["wdb"]
        params = kwargs["params"]
        metrics = kwargs["metrics"]

        for trial_params in params:
            # Create a KFold object with 10 folds
            kf = KFold(n_splits=10, shuffle=True)

            # Build the Keras model
            keras_model = self.__build_model(metrics=metrics, **trial_params)

            # Create an empty list to hold the table data
            table_data = []
            metrics_names = []

            for fold, (train_index, val_index) in enumerate(kf.split(self.features)):
                # Split the data into training and validation sets for the current fold
                x_train, x_val = self.features[train_index], self.features[val_index]
                y_train, y_val = self.labels[train_index], self.labels[val_index]
                # Fit the Keras model
                history = keras_model.fit(x_train[..., np.newaxis], self.categorize_labels(y_train),
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          validation_data=(x_val[..., np.newaxis], self.categorize_labels(y_val)),
                                          workers=6,
                                          use_multiprocessing=True,
                                          callbacks=[WandbCallback(save_model=False)])
                # Insert results of this fold into table_data
                if len(metrics_names) == 0:
                    metrics_names = [f"val_{metric}" for metric in keras_model.metrics_names]
                results = [fold] + [history.history[metric][-1] for metric in metrics_names]
                table_data.append(results)

            columns = ["Fold"] + metrics_names

            # Compute the mean values for each column
            mean_values = ["Mean"] + [np.mean([row[i] for row in table_data[1:]])
                                      for i in range(1, len(metrics_names) + 1)]

            # Append the mean values to the table data
            table_data.append(mean_values)

            # Create a Pandas DataFrame with the specified columns
            df = pd.DataFrame(table_data, columns=columns)

            # Create the wandb.Table with the specified columns
            table = wandb.Table(dataframe=df, columns=columns, allow_mixed_types=True)

            # Log the results to Weights & Biases
            wdb.log({"CV Results": table})
