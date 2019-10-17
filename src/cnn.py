import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical


class DigitCNN:

    def __init__(self, x, y):
        self.cnn = Sequential()
        self.x = x
        self.y = to_categorical(y)

    def build(self):
        self.cnn.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
        self.cnn.add(Dropout(0.5))
        self.cnn.add(Conv2D(32, kernel_size=3, activation="relu"))
        self.cnn.add(Flatten())
        self.cnn.add(Dense(16, activation="relu"))
        self.cnn.add(Dense(10, activation="softmax"))

    def train(self):
        self.cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.cnn.fit(self.x, self.y, epochs=5)

    def predict(self, to_pred):
        preds = self.cnn.predict(to_pred)
        preds = [np.where(pred == np.max(pred))[0][0] for pred in preds]
        return preds
