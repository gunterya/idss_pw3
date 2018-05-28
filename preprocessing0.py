import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def preprocessing(data_path):
    # read file
    d = np.genfromtxt(data_path, delimiter=',') # when return, '?' will be nan

    # delete missing data # TODO: missing data treatment
    d = d[~np.isnan(d).any(axis=1)] # remove rows which containing nan
    # print(d)

    # split X, Y
    X = d[:, 0:13].astype(float)
    Y = d[:, 13]
    Y[Y > 0] = 1 # shape: (:, 1)

    # normalize X
    # X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # print(X)

    # Y to be 2-class
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoder_Y = encoder.transform(Y)
    Y = np_utils.to_categorical(encoder_Y) # shape: (:, 2)

    return X, Y