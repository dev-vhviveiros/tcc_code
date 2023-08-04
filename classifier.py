import numpy as np
import wandb
from wandb.keras import WandbCallback
import pandas as pd
import datetime
from tensorflow.keras.metrics import AUC, Recall, Precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer, f1_score
from dataset_representation import DATASET_TAG, Model
from models import classifier_model
from utils import check_folder, load_config
from training_plot import TrainingPlot
from tuner import CustomTuner
from wandb_utils import WB_JOB_MODEL_FIT, WandbUtils
from keras_tuner.engine import tuner as tuner_module
import tensorflow.keras.backend as K


class Classifier:
    """
    Use: 
    input_file = your_file.csv when you want to use a new model
    import_model = if you saved a model and want to import it
    """

    def __init__(self, characteristics_artifact: str = None, model_artifact: str = None):
        self.wdb_project = load_config("wb_project_name")
        if characteristics_artifact is not None:
            self.__load_characteristics(characteristics_artifact)
        if model_artifact is not None:
            self.__import_model(model_artifact)

        print("Initializing classifier project: %s with:\n Characteristics: %s\nModel: %s" %
              (self.wdb_project, characteristics_artifact, model_artifact))

    def __load_characteristics(self, characteristics_artifact):
        """
        Loads the image characteristics and labels from a CSV file, splits the data into training and testing sets, and normalizes the image characteristics using the StandardScaler function.

        Args:
            characteristics_artifact (str): The path to the CSV file containing the image characteristics and labels.
            test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.2.
        Returns:
            None
        """
        # Read the CSV file containing the image characteristics
        characteristics_df = pd.read_csv(characteristics_artifact)
        print("SHAPE:" + str(characteristics_df.shape))

        # Extract the input data (image characteristics) and output data (labels)
        image_characteristics = characteristics_df.iloc[:, 0:267].values
        self.labels = characteristics_df.iloc[:, 268].values

        # Normalize the data using the StandardScaler function
        scaler = StandardScaler()
        self.norm_img_characteristics = scaler.fit_transform(image_characteristics)

    @staticmethod
    def custom_specificity(y_true, y_pred):
        """This code defines a function called custom_specificity that takes two parameters, y_true and y_pred. It then uses Keras backend functions to calculate the true negative (tn) and false positive (fp) values for the given labels of 0 and 1. Finally, it returns the ratio of true negatives to the sum of true negatives and false positives. This ratio is known as specificity, which measures how well a model can distinguish between classes."""
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        return tn / (tn + fp + K.epsilon())

    @staticmethod
    def custom_sensitivity(y_true, y_pred):
        """This code defines a function called custom_sensitivity, which takes two parameters, y_true and y_pred. It then uses Keras backend functions to calculate the false negatives (fn) and true positives (tp). Finally, it returns the ratio of true positives to the sum of true positives and false negatives. This ratio is a measure of sensitivity, which is the ability of a model to correctly identify positive cases."""
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        return tp / (tp + fn + K.epsilon())

    def tune(self, hypermodel, oracle, epochs: int, objective: str, batch_size_callout):
        tuner = CustomTuner(
            batch_size_callout=batch_size_callout,
            hypermodel=hypermodel,
            executions_per_trial=2,
            directory='tuner',
            overwrite=True,
            oracle=oracle
        )

        tuner.search(self.norm_img_characteristics, self.labels, epochs=epochs, objective=objective)

    def plot_confusion_matrix(self,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True,
                              save_dir=None):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                    the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                    see http://matplotlib.org/examples/color/colormaps_reference.html
                    plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                    If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                # sklearn.metrics.confusion_matrix
                            normalize    = True,                # show proportions
                            target_names = y_labels_vals,       # list of names of the classes
                            title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        pc = self.model.predict_classes(self.normalized_testing_image_characteristics)
        cm = confusion_matrix(pc, self.val_labels)
        target_names = ['Covid-19', 'Normal']

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
            accuracy, misclass))
        if save_dir is not None:
            plt.savefig(save_dir)
        plt.show()

    def predict(self, x):
        """This code defines a function called "predict" that takes in an argument "x" and returns the predicted classes of the given input. The function prints the predicted classes and then returns them. The prediction is done using the "model" attribute of the class."""
        pred = self.model.predict_classes(x)
        print(pred)
        return pred
