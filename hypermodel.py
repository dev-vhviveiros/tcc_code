from kerastuner import HyperModel
from tensorflow.keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop


class CustomHyperModel(HyperModel):
    """
    A custom hypermodel subclassed from `HyperModel` that defines the architecture of the neural network.
    """

    def __init__(self, optimizer_callout, activation_callout, activation_output_callout, dropout_callout, loss_callout, learning_rate_callout, dense_layers_callout, filters_callout, kernel_size_callout, pool_size_callout, conv_layers_callout, units_callout, use_same_units_callout, metrics=["accuracy"]):
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
        self.dense_layers_callout = dense_layers_callout
        self.conv_layers_callout = conv_layers_callout
        self.filters_callout = filters_callout
        self.kernel_size_callout = kernel_size_callout
        self.pool_size_callout = pool_size_callout
        self.units_callout = units_callout
        self.use_same_units_callout = use_same_units_callout
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
        dense_layers_hp = self.dense_layers_callout(hp)
        filters_hp = self.filters_callout(hp)
        kernel_size_hp = self.kernel_size_callout(hp)
        pool_size_hp = self.pool_size_callout(hp)
        conv_layers_hp = self.conv_layers_callout(hp)
        units_hp = self.units_callout(hp)
        use_same_units_hp = self.use_same_units_callout(hp)

        # Get the optimizer
        optimizer = self.get_optimizer(optimizer_hp, learning_rate_hp)

        # Create a Sequential model
        model = Sequential()

        # Add the specified number of convolutional layers to the model
        if (conv_layers_hp >= 1):
            for i in range(0, conv_layers_hp):
                layer_filter_size = (2 ** (i+1)) * filters_hp
                model.add(Conv1D(filters=layer_filter_size, kernel_size=kernel_size_hp,
                                 activation=activation_hp))
                model.add(MaxPooling1D(pool_size=pool_size_hp))

            # Add a Flatten layer to convert the output of the convolutional layers to a 1D tensor
            model.add(Flatten())

        # Calculate the minimum divisor to ensure the number of units in each dense layer is greater than 2
        divisor = 2.0
        while round(units_hp / (divisor ** dense_layers_hp)) <= 2:
            divisor -= 0.1

        # Add the specified number of dense layers to the model
        for i in range(0, dense_layers_hp):
            layer_units = units_hp
            if not use_same_units_hp:  # Use or not different values for each layer
                layer_units = units_hp / (divisor ** i)
                layer_units = max(2, round(layer_units))  # Adjust the layer_units to be at least 2
            if layer_units < 2:
                break
            model.add(Dense(units=layer_units, activation=activation_hp))
            model.add(Dropout(rate=dropout_hp))

        # Add the output layer
        units = 3 if activation_output_hp == 'softmax' else 1
        model.add(Dense(units=units, activation=activation_output_hp))

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
