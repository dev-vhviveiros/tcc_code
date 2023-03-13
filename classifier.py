import wandb
from wandb.keras import WandbCallback
import pandas as pd
import datetime
from tensorflow.keras.metrics import AUC, Recall, Precision
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer, f1_score
from models import classifier_model
from utils import WB_ARTIFACT_COVID_DATASET_TAG, WB_ARTIFACT_DATASET_TAG, WB_JOB_LOAD_ARTIFACTS, WB_JOB_MODEL_FIT, check_folder, load_config, model_path
from training_plot import TrainingPlot
from wandb_utils import WandbUtils


class Classifier:
    """
    Uso: 
    input_file = your_file.csv when you want to use a new model
    import_model = if you saved a model and want to import it
    """

    def __init__(self, input_file=None, import_model=None):
        self.project_name = load_config("wb_project_name")
        if input_file is not None:
            self.load_dataset(input_file)
        if import_model is not None:
            self.__import_model(import_model)

    def load_dataset(self, file, test_size=0.2):
        """This code is used to load a dataset from a csv file and split it into test and training sets. It also normalizes the data. 
                The function takes two parameters: 'file' which is the path to the csv file, and 'test_size' which is the size of the test set as a proportion of the total dataset. 
                The function reads in the csv file using pandas, then splits it into entries and results. The entries are split into X_train and X_test, while results are split into y_train and y_test. 
                Finally, the data is normalized using StandardScaler()."""
        # Read csv
        ctcs = pd.read_csv(file)
        entries = ctcs.iloc[:, 1:269].values
        results = ctcs.iloc[:, 269].values

        # Split into test and training
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            entries, results, test_size=test_size, random_state=0)

        # Normalize data
        sc = StandardScaler()
        self.X_train = sc.fit_transform(X_train)
        self.X_test = sc.transform(X_test)

    def log_artifacts(self):
        """This code uses the Wandb library to log artifacts. It initializes a run with a project name of "tcc" and job type of "load-artifacts". It creates two datasets, training_set and test_set, from self.X_train, self.y_train, self.X_test, and self.y_test. It then creates an artifact called covid_dataset with type dataset and description "Raw covid dataset, split into train/test". The metadata for this artifact includes the sizes of the training set and test set datasets. Finally, it logs the artifact and finishes the run."""
        with wandb.init(project=self.project_name, job_type=WB_JOB_LOAD_ARTIFACTS) as run:
            training_set = [self.X_train, self.y_train]
            test_set = [self.X_test, self.y_test]

            raw_data = wandb.Artifact(
                WB_ARTIFACT_COVID_DATASET_TAG, type=WB_ARTIFACT_DATASET_TAG,
                description="Raw covid dataset, split into train/test",
                metadata={"sizes": [len(dataset) for dataset in [training_set, test_set]]})

            run.log_artifact(raw_data)

            wandb.finish()

    def __format_validation(self, grid_cv):
        """This function takes in a grid_cv object as an argument and returns a dictionary containing the best results of the cross-validation for each metric. 
                The key_filter function filters out the keys in grid_cv.cv_results_ that start with 'split' and contain the name of a metric, then the dictionary is created by looping through each metric and adding its best result to the dictionary."""
        def key_filter(key):
            return list(filter(lambda x: x.startswith('split') and x.__contains__(key), grid_cv.cv_results_))

        return {metric: [grid_cv.cv_results_[m][grid_cv.best_index_] for m in key_filter(metric)]
                for metric in self.metrics}

    @staticmethod
    def custom_specificity(y_true, y_pred):
        """This code defines a function called custom_specificity that takes two parameters, y_true and y_pred. It then uses the confusion_matrix function to calculate the true negative (tn) and false positive (fp) values for the given labels of 0 and 1. Finally, it returns the ratio of true negatives to the sum of true negatives and false positives. This ratio is known as specificity, which measures how well a model can distinguish between classes."""
        tn, fp, _, _ = confusion_matrix(
            y_true, y_pred, labels=[0, 1]).ravel()
        return (tn / (tn + fp))

    @staticmethod
    def custom_sensitivity(y_true, y_pred):
        """This code defines a function called custom_sensitivity, which takes two parameters, y_true and y_pred. It then uses the confusion_matrix function to calculate the false negatives (fn) and true positives (tp). Finally, it returns the ratio of true positives to the sum of true positives and false negatives. This ratio is a measure of sensitivity, which is the ability of a model to correctly identify positive cases."""
        _, _, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]).ravel()
        return (tp / (tp + fn))

    def validation(self, n_jobs=-2, cv=10, batch_size=-1, epochs=-1, units=-1, optimizer=['adam'], activation=['relu'], activation_output=['sigmoid'], loss=['binary_crossentropy'], save_path=None):
        """This code is a function that uses the KerasClassifier class to perform a grid search using cross-validation (cv) and the given parameters. The parameters include batch size, epochs, units, optimizer, activation, activation output, and loss. The metrics used for scoring are accuracy, precision, f1_score, sensitivity, and specificity. The learning rate is set to 0.001 and WandbCallback is used as a callback for the grid search. If the save_path parameter is not None and both batch size and epochs have been specified, then the validation results are saved to the given path. Finally, the grid search results are returned."""

        classifier = KerasClassifier(build_fn=classifier_model)

        parameters = {'batch_size': batch_size,
                      'epochs': epochs,
                      'units': units,
                      'optimizer': optimizer,
                      'activation': activation,
                      'activationOutput': activation_output,
                      'loss': loss}

        self.metrics = {'accuracy': 'accuracy',
                        'precision': 'precision',
                        'f1_score': make_scorer(f1_score),
                        'sensitivity': make_scorer(Classifier.custom_sensitivity),
                        'specificity': make_scorer(Classifier.custom_specificity)}

        grid_search = GridSearchCV(estimator=classifier,
                                   verbose=2,
                                   param_grid=parameters,
                                   n_jobs=n_jobs,
                                   scoring=self.metrics,
                                   refit='accuracy',
                                   return_train_score=False,
                                   cv=cv)

        wandb.config = {
            "learning_rate": 0.001,
            "epochs": epochs,
            "batch_size": batch_size
        }

        grid_search.fit(self.X_train, self.y_train,
                        callbacks=[WandbCallback()])

        if save_path is not None and len(batch_size) + len(epochs) == 2:
            self.__save_validation(grid_search, save_path)

        return grid_search

    def __save_validation(self, grid_search, save_path):
        """This function saves the results of a grid search validation. It takes two parameters, the grid search object and the save path. It formats the results of the grid search into a dataframe and then saves it to a csv file at the specified save path."""
        result_set = self.__format_validation(grid_search)
        pd.DataFrame(result_set).to_csv(save_path)

    def fit(self, logs_folder, export_model=True, batch_size=16, epochs=300, units=180, optimizer='sgd', activation='relu', activation_output='sigmoid', loss='binary_crossentropy'):
        """This code is used to fit a classifier model with the given parameters. It initializes a run on Weights & Biases (Wandb) and logs the dataset generated for the model. It then creates a classifier model using the given parameters and fits it to the training data. The TrainingPlot and WandbCallback callbacks are used for visualizing the training process. Finally, if an export directory is provided, it exports the model to that directory and logs an artifact of the model on Wandb."""
        date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        wb_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "units": units,
            "optimizer": optimizer,
            "activation": activation,
            "activation_output": activation_output,
            "loss": loss
        }

        with (wandb.init(project=self.project_name, job_type=WB_JOB_MODEL_FIT, magic=True, name="fit__" + date_time,
                         config=wb_config)) as run:
            generated_dataset = wandb.Artifact(
                "characteristics", type=WB_ARTIFACT_DATASET_TAG)
            generated_dataset.add_file(
                "characteristics.csv")
            run.log_artifact(generated_dataset)
            print("\nExporting generated dataset...\n")

            # check_folder(logs_folder, False)
            # log_dir = logs_folder + date_time
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(
            #     log_dir=log_dir, histogram_freq=1)
            self.model = classifier_model(optimizer, activation, activation_output, units,
                                          ['accuracy', Precision(), AUC(), Recall()], loss)
            self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose=1, workers=12, use_multiprocessing=True,
                           validation_data=(self.X_test, self.y_test), callbacks=[TrainingPlot(epochs), WandbCallback(data_type="histogram")])
            if export_model:
                self.__export_model()

            run.log_artifact(WandbUtils.__generate_model_artifact())
            print("\nExporting model...\n")

    def __export_model(self):
        """This code is a function that exports a model. It takes two parameters: save_dir (the directory where the model should be saved) and date_time (the current date and time). 
The first line calls the check_folder() function to check if the save_dir exists, and creates it if it does not. 
The second line saves the model in the save_dir directory with the name "model.h5"."""
        check_folder(model_path(), False)
        #self.model.save(save_dir + 'save_' + date_time + '.h5')
        self.model.save(model_path())

    def __import_model(self, model_dir, optimizer='sgd', activation='relu', activation_output='sigmoid', loss='binary_crossentropy', units=180):
        """This function creates a classifier model with the given parameters and loads the weights from the given directory. 
Parameters: 
model_dir (string): The directory of the model weights to be loaded 
optimizer (string): The optimizer algorithm used for training 
activation (string): The activation function used in the model 
activation_output (string): The activation function used for the output layer 
loss (string): The loss function used for training 
units (integer): Number of units in the hidden layers 
Returns: self.model (object)"""
        self.model = classifier_model(optimizer, activation, activation_output, units,
                                      ['accuracy', Precision(), AUC(), Recall()], loss)
        self.model.load_weights(model_dir)
        return self.model

    def __confusion_matrix(self):
        """This code uses the model attribute from the instance to predict classes from the X_test attribute of the instance. It then uses the confusion_matrix function to compare the predicted classes with the y_test attribute of the instance and returns the matrix."""
        pred = self.model.predict_classes(self.X_test)
        matrix = confusion_matrix(pred, self.y_test)
        return (matrix)

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

        cm = self.__confusion_matrix()
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
