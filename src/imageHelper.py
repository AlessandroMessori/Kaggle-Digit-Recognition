import numpy as np
import matplotlib.pyplot as plt


def parse_images(values):
    data = np.zeros((len(values), 28, 28, 1), dtype=np.uint8)

    for (i, row) in enumerate(values):
        for (j, col) in enumerate(row):
            data[i, int(j / 28), j % 28, 0] = col

    return data


def show_images(images, cols=1, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
