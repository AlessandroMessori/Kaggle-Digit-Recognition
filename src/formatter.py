import numpy as np
import pandas as pd


def save_preds(preds):
    output = {
        'ImageId': np.arange(len(preds)) + 1,
        'Label': np.array(preds, dtype='int64')
    }

    output = pd.DataFrame.from_dict(output)
    output.set_index(output.columns[1])
    output = output.iloc[:, [0, 1]]

    output.to_csv(path_or_buf='./data/preds.csv', index=False)
