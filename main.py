import pandas as pd
from PIL import Image
from src.imageHelper import parse_images, show_images
from src.cnn import DigitCNN

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

x = train.iloc[:, 1:]
y = train.iloc[:, 0]

classifier = DigitCNN()
classifier.build()

dataset_images = parse_images(x.values)

pil_images = [Image.fromarray(img) for img in dataset_images[0:10]]
show_images(pil_images)
