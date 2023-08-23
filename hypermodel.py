from kerastuner import HyperModel
from tensorflow.keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop


class CustomHyperModel(HyperModel):
    """
    A custom hypermodel subclassed from `HyperModel` that defines the architecture of the neural network.
    """

    def __init__(self, optimizer_callout, activation_callout, activation_output_callout, dropout_callout, loss_callout, learning_rate_callout, num_layers_callout, filters_callout, kernel_size_callout, pool_size_callout, metrics=["accuracy"]):
        """
        Initializes a new instance of the CustomHyperModel class.

        Args:
            optimizer_callout: A callable that takes a `HyperParameters` object and returns an optimizer.
            activation_callout: A callable that takes a `HyperParameters` object and returns an activation function for the hidden layers.
            activation_output_callout: A callable that takes a `HyperParameters` object and returns an activation function for the output layer.
            dropout_callout: A callable that takes a `HyperParameters` object and returns the dropout rate for the hidden layers.
            loss_callout: A callable that takes a `HyperParameters` object and returns the loss function.
            learning_rate_callout: A callable that takes a `HyperParameters` object and returns the learning rate for the optimizer.
            num_layers_callout: A callable that takes a `HyperParameters` object and returns the number of convolutional layers to use.
            filters_callout: A callable that takes a `HyperParameters` object and returns the number of filters to use in the convolutional layers.
            kernel_size_callout: A callable that takes a `HyperParameters` object and returns the kernel size to use in the convolutional layers.
            pool_size_callout: A callable that takes a `HyperParameters` object and returns the pool size to use in the max pooling layers.
            metrics: A list of metrics to evaluate the model with during training and testing.
        """
        self.optimizer_callout = optimizer_callout
        self.activation_callout = activation_callout
        self.activation_output_callout = activation_output_callout
        self.dropout_callout = dropout_callout
        self.loss_callout = loss_callout
        self.learning_rate_callout = learning_rate_callout
        self.num_layers_callout = num_layers_callout
        self.filters_callout = filters_callout
        self.kernel_size_callout = kernel_size_callout
        self.pool_size_callout = pool_size_callout
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
        num_layers_hp = self.num_layers_callout(hp)
        filters_hp = self.filters_callout(hp)
        kernel_size_hp = self.kernel_size_callout(hp)
        pool_size_hp = self.pool_size_callout(hp)

        # Get the optimizer
        optimizer = self.get_optimizer(optimizer_hp, learning_rate_hp)

        model = Sequential([
            Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(348, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

        # # Create the model
        # model = Sequential()

        # # Add the layers
        # model.add(Conv1D(filters=filters_hp, kernel_size=kernel_size_hp, input_shape=(348, 1),
        #                  activation=activation_hp))
        # model.add(MaxPooling1D(pool_size=pool_size_hp))
        # model.add(Dropout(rate=dropout_hp))
        # for _ in range(num_layers_hp):
        #     model.add(Conv1D(filters=filters_hp, kernel_size=kernel_size_hp,
        #                      activation=activation_hp, padding='same'))
        #     model.add(MaxPooling1D(pool_size=pool_size_hp))
        #     model.add(Dropout(rate=dropout_hp))

        # # Add the output layer
        # model.add(Flatten())
        # model.add(Dense(units=1, activation=activation_output_hp))

        # Compile the model with the given optimizer, loss function, and metrics
        model.compile(optimizer=optimizer, loss=loss_hp, metrics=self.metrics)

        return model

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
            "rmsprop": RMSprop
        }

        if optimizer_name not in optimizers:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")
        optimizer_class = optimizers[optimizer_name]
        return optimizer_class(learning_rate)
