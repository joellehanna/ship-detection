from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16

from models.classifier import Classifier


class VGG(keras.Model, Classifier):
    r"""
    """
    def __init__(self, name='name'):
        super(VGG, self).__init__(name=name)
        self.base_model = VGG16(input_shape=(80, 80, 3),
                                include_top=False,
                                weights='imagenet')
        self.flatten1 = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(512, activation='relu')
        self.dropout1 = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        """
        Computes the forward propagation.
        """
        for layer in self.base_model.layers:
            layer.trainable = False
        x = self.base_model(inputs)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        return self.dense2(x)
