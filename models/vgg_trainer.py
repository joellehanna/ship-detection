import os

import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from models.vgg import VGG

from helper import sliding_window, draw_rois, load_images_from_folder, to_num


class VGGTrainer(VGG):
    """
    """
    def __init__(self, name='name'):
        super(VGGTrainer, self).__init__(name=name)
        self.class_weights = None
        self.X_mean = None
        self.X_std = None
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
        history = self.fit(self.X_train, self.y_train.reshape(-1, 1), epochs=2,
                           callbacks=[es], class_weight=self.class_weights,
                           validation_data=(self.X_val, self.y_val.reshape(-1, 1)))

        test_score = self.evaluate(self.X_test, self.y_test)
        print(test_score)

        pd.DataFrame(history.history).plot(figsize=(16, 10))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.savefig('outputs/vgg_plots.png')

    def create_dataset(self):
        path_to_dataset = os.path.join('.', 'dataset', 'shipsnet', 'shipsnet')
        images, df = load_images_from_folder(path_to_dataset)

        X = np.asarray(images)
        df.label = df.label.apply(to_num)
        y = np.asarray(list(df.label))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1)

        self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(self.y_train),
                                            self.y_train)))
        # Scale data
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.X_val = np.array(self.X_val)

        self.X_mean = self.X_train.mean(axis=0, keepdims=True)
        self.X_std = self.X_train.std(axis=0, keepdims=True) + 1e-7
        self.X_train = (self.X_train - self.X_mean) / self.X_std
        self.X_val = (self.X_val - self.X_mean) / self.X_std
        self.X_test = (self.X_test - self.X_mean) / self.X_std

    def detect_ships(self, image):
        x_to_keep = []
        y_to_keep = []

        (winW, winH) = (80, 80)

        for (x, y, window) in sliding_window(image, stepSize=20, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            window = (window - self.X_mean) / self.X_std
            pred = np.round(self.predict(window))
            if pred:
                print('Ship detected!')
                x_to_keep.append(x)
                y_to_keep.append(y)

        img_draw = draw_rois(image, winW, winH, np.asarray([x_to_keep, y_to_keep]), color=(255, 0, 0))
        plt.figure(figsize=(12, 12))
        plt.imsave('outputs/vgg.png', img_draw)
