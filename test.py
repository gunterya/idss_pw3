'''
Integration
'''
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.models import load_model
from faphy import get_weights
from preprocessing import preprocessing
from ann import ANN

seed = 2018
np.random.seed(seed)

# parameters
TRAIN_AGAIN = False
WEIGHT_AGAIN = False
DATA_PATH = 'data/processed_data.csv'
PAIRWISE_PATH = '' # TODO: do it need to change to file and then input?!
W_PATH = 'data/weights'
MODEL_PATH = 'data/ANNmodel.h5'

pred_data = '' # new data for prediction
# TODO: ADD time consume analysis of each block


if __name__ == '__main__':
    #############################
    #  Load data                #
    #############################
    X, y = preprocessing(DATA_PATH)
    print('X shape:', X.shape)
    print('Y shape:', y.shape)

    # split train(60%)+validation(20%)=train(80%) /test(20%)
    # I split train/validation in ANN model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    #############################
    #  Load weights (faphy)     #
    #############################
    if os.path.isfile(W_PATH) & (not WEIGHT_AGAIN):
        print('\nLoading weights...')
        w = np.fromfile('data/weights', dtype=np.float128, sep=' ')
        w = np.reshape(w, len(w))
    else:
        print('\nComputing weights...')
        w = get_weights()
    print(w)
    # paper_w = np.array([[0.0822, 0.0287, 0.1333, 0.0645, 0.0559, 0.0531, 0.0452, 0.1235, 0.0696, 0.0997, 0.0386, 0.0849, 0.1708]])

    # X * attribute_weight
    X_train = np.multiply(X_train, w)
    X_test = np.multiply(X_test, w)


    #############################
    #  ANN                      #
    #############################
    if (not os.path.isfile(MODEL_PATH)) | TRAIN_AGAIN | WEIGHT_AGAIN:
        print('\nTraing ANN(10-13-2)...')
        model = ANN(X_train, y_train, MODEL_PATH)
    else:
        print('\nLoading ANN(10-13-2)...')
        model = load_model(MODEL_PATH)

    # evaluate model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Test %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # predict (new input)
    # X = ''
    # predictions = model.predict(X)


    K.clear_session()  # delete session from keras backend
