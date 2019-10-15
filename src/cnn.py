from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


class DigitCNN:

    def __init__(self):
        self.cnn = Sequential()

    def build(self):
        self.cnn.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
        self.cnn.add(Conv2D(32, kernel_size=3, activation="relu"))
        self.cnn.add(Flatten())
        self.cnn.add(Dense(10, activation="softmax"))
