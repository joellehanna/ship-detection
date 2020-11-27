from tensorflow import keras

from models.classifier import Classifier


class CNN(keras.Model, Classifier):
    r"""
    """
    def __init__(self, name='name'):
        super(CNN, self).__init__(name=name)
        self.input1 = keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[80, 80, 3])
        self.maxpool1 = keras.layers.MaxPooling2D(2)
        self.conv1 = keras.layers.Conv2D(128, 3, activation="relu", padding="same")
        self.conv2 = keras.layers.Conv2D(128, 3, activation="relu", padding="same")
        self.maxpool2 = keras.layers.MaxPooling2D(2)
        self.conv3 = keras.layers.Conv2D(256, 3, activation="relu", padding="same")
        self.conv4 = keras.layers.Conv2D(256, 3, activation="relu", padding="same")
        self.maxpool3 = keras.layers.MaxPooling2D(2)
        self.conv5 = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(128, activation="relu")
        self.dropout1 = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(64, activation="relu")
        self.dropout2 = keras.layers.Dropout(0.5)
        self.dense3 = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        """
        Computes the forward propagation.
        """
        x = self.input1(inputs)
        x = self.maxpool1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool3(x)
        x = self.conv5(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.dense3(x)
