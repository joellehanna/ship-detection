import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from models.fully_connected import FullyConnected

from helper import sliding_window, draw_rois


class FullyConnectedTrainer(FullyConnected):
    """
    """
    def __init__(self, name='name'):
        super(FullyConnectedTrainer, self).__init__(name=name)
        self.scaler = None
        self.class_weights = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def train(self):
        """
        Trains the model.
        """
        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=5,
                                           mode='auto',
                                           baseline=None, restore_best_weights=True)
        self.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        history = self.fit(np.expand_dims(self.X_train, axis=-1),
                           np.expand_dims(self.y_train, axis=-1),
                           validation_data=(np.expand_dims(self.X_val, axis=-1), np.expand_dims(self.y_val, axis=-1)),
                           callbacks=[es],
                           class_weight=self.class_weights,
                           epochs=50)

        test_score = self.evaluate(np.expand_dims(self.X_test, axis=-1), np.expand_dims(self.y_test, axis=-1))
        print(test_score)

        pd.DataFrame(history.history).plot(figsize=(16, 10))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.savefig('outputs/fullyconnected_plots.png')

    def create_dataset(self):
        df = pd.read_json('dataset/shipsnet.json')

        # Create train test val data
        X = np.asarray(list(df.data))
        y = np.asarray(list(df.labels))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1)
        self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(self.y_train),
                                            self.y_train)))
        # Scale data
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

    def detect_ships(self, image):
        x_to_keep = []
        y_to_keep = []

        (winW, winH) = (80, 80)

        for (x, y, window) in sliding_window(image, stepSize=40, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            window_reshaped = np.concatenate((np.array(window[:, :, 0]).reshape(-1, 1),
                                              np.array(window[:, :, 1]).reshape(-1, 1),
                                              np.array(window[:, :, 2]).reshape(-1, 1)), axis=0)
            window_reshaped = self.scaler.transform(window_reshaped.reshape(1, -1))
            pred = np.round(self.predict(np.expand_dims(np.array(window_reshaped).reshape(1, -1), axis=-1)))
            if pred:
                print('Ship detected!')
                x_to_keep.append(x)
                y_to_keep.append(y)

        img_draw = draw_rois(image, winW, winH, np.asarray([x_to_keep, y_to_keep]), color=(255, 0, 0))
        plt.figure(figsize=(12, 12))
        plt.imsave('outputs/fullyconnected.png', img_draw)
