from kerastuner import HyperModel
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta


class CustomHyperModel(HyperModel):
    """
    A custom hypermodel subclassed from `HyperModel` that defines the architecture of the neural network.
    """

    def __init__(self, optimizer_callout, activation_callout, activation_output_callout, units_callout, dropout_callout, loss_callout, learning_rate_callout, num_layers_callout, metrics=["accuracy"]):
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
        self.learning_rate_callout = learning_rate_callout
        self.num_layers_callout = num_layers_callout
        self.metrics = metrics

    def build(self, hp):
        """
        Builds a Keras model with the specified hyperparameters.

        Args:
            hp (HyperParameters): A HyperParameters object containing the hyperparameters to use.

        Returns:
            A compiled Keras model with the specified hyperparameters.
        """
        # Get the hyperparameters
        optimizer_hp = self.optimizer_callout(hp)
        learning_rate_hp = self.learning_rate_callout(hp)
        activation_hp = self.activation_callout(hp)
        activation_output_hp = self.activation_output_callout(hp)
        loss_hp = self.loss_callout(hp)
        dropout_hp = self.dropout_callout(hp)
        units_hp = self.units_callout(hp)
        num_layers_hp = self.num_layers_callout(hp)

        # Get the optimizer
        optimizer = self.get_optimizer(optimizer_hp, learning_rate_hp)

        # Create the model
        classifier = Sequential()

        # Add the layers
        classifier.add(Dense(units=units_hp, activation=activation_hp, input_shape=(348,)))
        classifier.add(Dropout(rate=dropout_hp))
        for _ in range(num_layers_hp):
            classifier.add(Dense(units=units_hp, activation=activation_hp))
            classifier.add(Dropout(rate=dropout_hp))

        # Add the output layer
        classifier.add(Dense(units=1, activation=activation_output_hp))

        # Compile the model with the given optimizer, loss function, and metrics
        classifier.compile(optimizer=optimizer, loss=loss_hp, metrics=self.metrics)

        return classifier

    def get_optimizer(self, optimizer_name, learning_rate):
        """
        Returns an instance of an optimizer with the specified name and learning rate.

        Args:
            optimizer_name (str): The name of the optimizer to use.
            learning_rate (float): The learning rate to use for the optimizer.

        Returns:
            An instance of the specified optimizer with the specified learning rate.
        """
        optimizers = {
            "sgd": SGD,
            "adam": Adam,
            "rmsprop": RMSprop,
            "adagrad": Adagrad,
            "adadelta": Adadelta
        }

        if optimizer_name not in optimizers:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")
        optimizer_class = optimizers[optimizer_name]
        return optimizer_class(learning_rate)
