import pandas as pd
from src.imageHelper import parse_images
from src.cnn import DigitCNN
from src.formatter import save_preds

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

x_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]

train_images = parse_images(x_train.values)
test_images = parse_images(test.values)

classifier = DigitCNN(train_images, y_train.values)
classifier.build()
classifier.train()

preds = classifier.predict(test_images)

save_preds(preds)
