from kerastuner import HyperModel, HyperParameters
from tensorflow.keras.layers import Dropout, Dense, ReLU
from tensorflow.keras.models import Sequential
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


class CNNHyperModel(HyperModel):
    def __init__(self, batch_size_callout, optimizer_callout, activation_callout, activation_output_callout, units_callout, dropout_callout, loss_callout, metrics=["accuracy"]):
        self.batch_size_callout = batch_size_callout
        self.optimizer_callout = optimizer_callout
        self.activation_callout = activation_callout
        self.activation_output_callout = activation_output_callout
        self.units_callout = units_callout
        self.dropout_callout = dropout_callout
        self.loss_callout = loss_callout
        self.metrics = metrics

    def build(self, hp: HyperParameters):
        optimizer_hp = self.optimizer_callout(hp)
        activation_hp = self.activation_callout(hp)
        activation_output_hp = self.activation_output_callout(hp)
        units_hp = self.units_callout(hp)
        dropout_hp = self.dropout_callout(hp)
        loss_hp = self.loss_callout(hp)
        
        model = Sequential()
        model.add(Dense(units=units_hp))
        model.add(
            Dense(units=units_hp, activation=activation_hp, input_shape=(268,)))
        model.add(Dropout(rate=dropout_hp))
        model.add(Dense(units=units_hp, activation=activation_hp))
        model.add(Dropout(rate=dropout_hp))
        model.add(Dense(units=units_hp, activation=activation_hp))
        model.add(Dropout(rate=dropout_hp))
        model.add(Dense(units=units_hp, activation=activation_hp))
        model.add(Dropout(rate=dropout_hp))
        model.add(Dense(units=units_hp, activation=activation_hp))
        model.add(Dropout(rate=dropout_hp))
        model.add(Dense(units=units_hp, activation=activation_hp))
        model.add(Dropout(rate=dropout_hp))
        model.add(Dense(units=1, activation=activation_output_hp))
        model.add(ReLU())
        model.compile(optimizer=optimizer_hp,
                      loss=loss_hp, metrics=self.metrics)
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.batch_size_callout(hp),
            workers=6,
            use_multiprocessing=True,
            **kwargs,
        )