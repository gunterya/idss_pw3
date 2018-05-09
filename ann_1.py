'''
writen by Keras
'''
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

def ANN(X_train, y_train, model_path, isPlot=0, isShow=0):
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=13, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    # compile model
    # from keras import optimizers
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # TODO: make sure which optimizer is better here
    # model.summary()

    # fit model
    stop_early = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    history = model.fit(X_train, y_train, # TODO: make sure how much batch_size is better here
              epochs=2000,
              verbose=isShow,
              validation_split= 20/85)  # train(65%)+validation(20%)=train(85%) /test(15%), split train/validation here
              # callbacks=[stop_early])

    if isPlot!=0:
        # print(history.history.keys()) # list all data in history
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(isPlot +'acc.png')
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