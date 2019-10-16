import numpy as np
import pandas as pd
from PIL import Image
from src.imageHelper import parse_images, show_images
from src.cnn import DigitCNN

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

output = {
    'ImageId': np.arange(len(preds)) + 1,
    'Label': np.array(preds, dtype='int64')
}

# print(len(np.arange(len(preds)) + 1))
# print(len(np.array(preds, dtype='int64')))

output = pd.DataFrame.from_dict(output)
output.set_index(output.columns[1])
output = output.iloc[:, [0, 1]]

output.to_csv(path_or_buf='./data/preds.csv', index=False)

'''pil_images = [Image.fromarray(img) for img in dataset_images[0:10]]
show_images(pil_images)'''
