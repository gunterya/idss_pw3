'''
writen by Keras
'''
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

def ANN(X_train, y_train, model_path, isShow=0):
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=13, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # TODO: make sure which optimizer is better here
    # model.summary()

    # fit model
    stop_early = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    history = model.fit(X_train, y_train, # TODO: make sure how much batch_size is better here
              epochs=2000,
              verbose=isShow,
              validation_split= 0.8/5)  # train(60%)+validation(20%)=train(80%) /test(20%), split train/validation here
              # callbacks=[stop_early])

    if isShow:
        # print(history.history.keys()) # list all data in history
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()

    # save model
    model.save(model_path)

    return model