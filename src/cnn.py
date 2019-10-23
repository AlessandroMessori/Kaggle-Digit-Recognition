import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical


class DigitCNN:

    def __init__(self, x, y):
        self.cnn = Sequential()
        self.x = x
        self.y = to_categorical(y)

    def build(self):
        self.cnn.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
        self.cnn.add(Conv2D(32, kernel_size=3, activation="relu"))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Dropout(0.28))
        self.cnn.add(Flatten())
        self.cnn.add(Dense(64, activation="relu"))
        self.cnn.add(BatchNormalization())
        self.cnn.add(Dropout(0.3))
        self.cnn.add(Dense(128, activation="relu"))
        self.cnn.add(BatchNormalization())
        self.cnn.add(Dropout(0.32))
        self.cnn.add(Dense(10, activation="softmax"))

    def train(self):
        self.cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.cnn.fit(self.x, self.y, epochs=25)

    def predict(self, to_pred):
        preds = self.cnn.predict(to_pred)
        preds = [np.where(pred == np.max(pred))[0][0] for pred in preds]
        return preds
