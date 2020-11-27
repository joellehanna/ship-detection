from tensorflow import keras

from models.classifier import Classifier


class FullyConnected(keras.Model, Classifier):
    r"""
    """
    def __init__(self, name='name'):
        super(FullyConnected, self).__init__(name=name)
        self.input1 = keras.layers.Flatten(input_shape=[19200, 1])
        self.dense1 = keras.layers.Dense(300, activation="relu")
        self.dense2 = keras.layers.Dense(100, activation="relu")
        self.dense3 = keras.layers.Dense(100, activation="relu")
        self.dense4 = keras.layers.Dense(100, activation="relu")
        self.dense5 = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        """
        Computes the forward propagation.
        :return: None
        """
        x = self.input1(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)
