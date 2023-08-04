from kerastuner import HyperModel, HyperParameters
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Set GPU memory growth to True
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CustomHyperModel(HyperModel):
    """
    A custom hypermodel subclassed from `HyperModel` that defines the architecture of the neural network.
    """

    def __init__(self, optimizer_callout, activation_callout, activation_output_callout, units_callout, dropout_callout, loss_callout, metrics=["accuracy"]):
        """
        Initializes a new instance of the CustomHyperModel class.

        Args:
            optimizer_callout: A callable that takes a `HyperParameters` object and returns an optimizer.
            activation_callout: A callable that takes a `HyperParameters` object and returns an activation function for the hidden layers.
            activation_output_callout: A callable that takes a `HyperParameters` object and returns an activation function for the output layer.
            units_callout: A callable that takes a `HyperParameters` object and returns the number of units for the hidden layers.
            dropout_callout: A callable that takes a `HyperParameters` object and returns the dropout rate for the hidden layers.
            loss_callout: A callable that takes a `HyperParameters` object and returns the loss function.
            metrics: A list of metrics to evaluate the model with during training and testing.
        """
        self.optimizer_callout = optimizer_callout
        self.activation_callout = activation_callout
        self.activation_output_callout = activation_output_callout
        self.units_callout = units_callout
        self.dropout_callout = dropout_callout
        self.loss_callout = loss_callout
        self.metrics = metrics

    def build(self, hp: HyperParameters):
        """
        Builds the neural network with the given hyperparameters.

        Args:
            hp: A `HyperParameters` object representing the hyperparameters for the neural network.

        Returns:
            A compiled `Sequential` model.
        """
        # Save the hyperparameters for later use
        self.hyperparameters = hp

        # Get the hyperparameters for the optimizer, activation functions, units, dropout rate, and loss function
        optimizer_hp = self.optimizer_callout(hp)
        activation_hp = self.activation_callout(hp)
        activation_output_hp = self.activation_output_callout(hp)
        units_hp = self.units_callout(hp)
        dropout_hp = self.dropout_callout(hp)
        loss_hp = self.loss_callout(hp)

        # Define the architecture of the neural network
        classifier = Sequential()
        classifier.add(Dense(units=units_hp, activation=activation_hp, input_shape=(267,)))
        classifier.add(Dropout(rate=dropout_hp))
        classifier.add(Dense(units=units_hp, activation=activation_hp))
        classifier.add(Dropout(rate=dropout_hp))
        classifier.add(Dense(units=units_hp, activation=activation_hp))
        classifier.add(Dropout(rate=dropout_hp))
        classifier.add(Dense(units=units_hp, activation=activation_hp))
        classifier.add(Dropout(rate=dropout_hp))
        classifier.add(Dense(units=units_hp, activation=activation_hp))
        classifier.add(Dropout(rate=dropout_hp))
        classifier.add(Dense(units=units_hp, activation=activation_hp))
        classifier.add(Dropout(rate=dropout_hp))
        classifier.add(Dense(units=1, activation=activation_output_hp))

        # Compile the model with the given optimizer, loss function, and metrics
        classifier.compile(optimizer=optimizer_hp, loss=loss_hp, metrics=self.metrics)

        return classifier
