'''
13-13-10-2
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from faphy import get_weights

W_PATH = 'data/weights'
W_AGAIN = False

def ANN(X_train, y_train, isW, w_trainable, model_path, isPlot=0, isShow=0):
    if isW:
        # attribute weight (faphy)
        if os.path.isfile(W_PATH) & (not W_AGAIN): # exist
            print('Loading attribute weights...')
            w = np.fromfile(W_PATH, dtype=np.float128, sep=' ')
            w = np.reshape(w, len(w))
        else: # not exist
            print('Computing attribute weights...')
            w = get_weights(W_PATH)
        # print(w)
        # build ANN layer1 weights
        W = np.zeros([13, 13])
        for i in range(13): W[i, i] = w[i]
        # print(W)
        b = np.zeros([13])

    # create model
    model = Sequential()
    if isW: model.add(Dense(13, input_dim=13, weights=[W, b], trainable=w_trainable)) # layer1 : 13-13 (attribute weight)
    model.add(Dense(10, input_dim=13, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])  # TODO: make sure which optimizer is better here
    # model.summary()

    # fit model
    history = model.fit(X_train, y_train,  # TODO: set these parameters in test.py
                        batch_size=50,
                        epochs=2000,
                        verbose=isShow,
                        validation_split=20 / 85)  # train(65%)+validation(20%)=train(85%) /test(15%), split train/validation here

    if isPlot != 0:
        # print(history.history.keys()) # list all data in history
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(isPlot + 'acc.png')
        # plt.show()
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(isPlot + 'loss.png')
        # plt.show()
        plt.close()

    # save model
    model.save(model_path)

    return model