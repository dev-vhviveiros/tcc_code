from kerastuner import HyperModel, HyperParameters
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


class CNNHyperModel(HyperModel):
    def __init__(self, optimizer: list, activation: list, activation_output: list, units: list, dropout:list=[0.2], metrics: list = ['accuracy'], loss=list):
        self.optimizer = optimizer
        self.activation = activation
        self.activation_output = activation_output
        self.units = units
        self.dropout = dropout
        self.metrics = metrics
        self.loss = loss

    def search_space(self, hp: HyperParameters):
        optimizer_hp = hp.Choice(
            "optimizer",
            self.optimizer,
            default=self.optimizer[0]
        )
        
        activation_hp = hp.Choice(
            "activation",
            self.activation,
            default=self.activation[0]
        )
        
        activation_output_hp = hp.Choice(
            "activation_output",
            self.activation_output,
            default=self.activation_output[0]
        )

        units_hp = hp.Int(
            "units",
            min_value=min(self.units),
            max_value=max(self.units),
            default=self.units[0]
        )
        
        dropout_hp = hp.Float(
            "dropout",
            min_value=min(self.dropout),
            max_value=max(self.dropout),
            default=self.dropout[0]
        )
        
        loss_hp = hp.Choice(
            "loss",
            self.loss,
            default=self.loss[0]
        )
        
        hyperparameters = {
            "optimizer": optimizer_hp,
            "activation": activation_hp,
            "activation_output": activation_output_hp,
            "units": units_hp,
            "dropout": dropout_hp,
            "loss" : loss_hp
        }

        return hyperparameters
        

    def build(self, hp: HyperParameters):
        build_hps = self.search_space(hp)
        model = Sequential()
        model.add(Dense(units=build_hps["units"]))
        model.add(
            Dense(units=build_hps["units"], activation=build_hps["activation"], input_shape=(268,)))
        model.add(Dropout(rate=build_hps["dropout"]))
        model.add(Dense(units=build_hps["units"], activation=build_hps["activation"]))
        model.add(Dropout(rate=build_hps["dropout"]))
        model.add(Dense(units=build_hps["units"], activation=build_hps["activation"]))
        model.add(Dropout(rate=build_hps["dropout"]))
        model.add(Dense(units=build_hps["units"], activation=build_hps["activation"]))
        model.add(Dropout(rate=build_hps["dropout"]))
        model.add(Dense(units=build_hps["units"], activation=build_hps["activation"]))
        model.add(Dropout(rate=build_hps["dropout"]))
        model.add(Dense(units=build_hps["units"], activation=build_hps["activation"]))
        model.add(Dropout(rate=build_hps["dropout"]))
        model.add(Dense(units=1, activation=build_hps["activation_output"]))
        model.compile(optimizer=build_hps["optimizer"],
                      loss=build_hps["loss"], metrics=self.metrics)
        return model
