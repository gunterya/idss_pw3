'''
Integration
'''
import os
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.models import load_model
from preprocessing import load_data

seed = 1
np.random.seed(seed)

# parameters
os.chdir('/Users/isabellepolizzi/Desktop/UPC/IDSS/PW3/idss_pw3/')
OUTPUT_DIR = 'results/'
DATA_DIR = 'data/'
DATA_PATH = DATA_DIR + 'processed_data.csv'
IMP_M = 'x' # missing data treatment
SCALE_M = 'min_max' # scale x(attributes)
isW = 1
W_TRAINABLE = 0
ANN_PATH = OUTPUT_DIR + 'ANNmodel'+'-IMP'+IMP_M+'-SCALE'+SCALE_M+'-isW'+str(isW)+'-wTrain'+str(W_TRAINABLE)+'.h5'
ANN_AGAIN = 0
isEval = 0
COLS_PATH = DATA_DIR + 'missing_cols' # missing cols



def train():
    # load data
    X, y = load_data(DATA_PATH, IMP_M, SCALE_M)

    # split train/valid/test
    print('\nSplit... (65% train, 20% valid, 15% test)')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    print('Number of training+validation data:', X_train.shape[0])
    print('Number of testing data:', X_test.shape[0])

    # model
    if (not os.path.isfile(ANN_PATH)) | ANN_AGAIN :
        print('\nTraining ANN...')
        from model import ANN
        model = ANN(X_train, y_train, isW, W_TRAINABLE, ANN_PATH, isPlot=OUTPUT_DIR)
    else:
        print('\nLoading ANN...')
        model = load_model(ANN_PATH)

    # evaluate model
    train_scores = model.evaluate(X_train, y_train, verbose=0)
    print("Train %s: %.2f%%" % (model.metrics_names[1], train_scores[1] * 100))
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    print("Test %s: %.2f%%" % (model.metrics_names[1], test_scores[1] * 100))

    # evaluation
    if isEval:
        print('\nEvaluating...')
        from eval import eval
        eval(model, X_test, y_test, OUTPUT_DIR)

def predict_HFp(X, needScale=True):
    # load model
    if (not os.path.isfile(ANN_PATH)) | ANN_AGAIN : train()
    model = load_model(ANN_PATH)

    if needScale: X = col_scaling(X)

    # prediction
    np.set_printoptions(precision=2)
    pred = model.predict(X)[:, 1] * 100
    return pred

def col_scaling(X):
    import pickle

    # scaler1 : handle cols(-missing_cols)
    cols = [i for i in range(X.shape[1])]
    missing_cols = np.fromfile(COLS_PATH, dtype=np.float128, sep=' ')
    missing_cols = np.reshape(missing_cols, len(missing_cols))
    for i in sorted(missing_cols, reverse=True):
        cols = np.delete(cols, i)
    scaler1 = pickle.load(open(DATA_DIR+'scaler1.sav', 'rb'))
    X[:, cols] = scaler1.transform(X[:, cols])

    # scaler2
    scaler2 = pickle.load(open(DATA_DIR+'scaler2.sav', 'rb'))
    X_scaled = scaler2.transform(X)
    return X_scaled


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print('Saving on... ' + os.getcwd() + '/'+ OUTPUT_DIR)


    # # predict original data
    # X, y = load_data(DATA_PATH, IMP_M, SCALE_M)
    # print(predict_HFp(X, needScale=False))

    # predict new data
    #new_x = np.array([[67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0]], dtype=float)
    #print('\nnew patient:', new_x)
    #print(predict_HFp(new_x))


    K.clear_session()  # delete session from keras backend
