import numpy as np
import pandas as pd
import keras
import sklearn
from keras.models import Model, Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from keras.layers import LSTM
from keras.layers import Input, concatenate
from ArgumentationClassifier import DataLoader as dl
import matplotlib.pyplot as plt


def train_RNN_classifier(dataset, epochs, singlePrint=False):

    print("-----TRAIN CLASSIFIER-----")

    x_train, y_train = dl.prepare_data_for_RNN(dataset)

    numberOfClasses = y_train.shape[1]

    lstm_input_dim = x_train["sentence1"].shape[1:]
    concatenateInput = x_train["sharedFeatures"].shape[1:]

    sentence1 = Input(lstm_input_dim, name="sentence1")
    sentence2 = Input(lstm_input_dim, name="sentence2")
    sharedFeatures = Input(concatenateInput, name="sharedFeatures")

    lstm1 = LSTM(16, return_sequences=False)(sentence1)
    lstm2 = LSTM(16, return_sequences=False)(sentence2)
    concatenateLayer = concatenate([lstm1, lstm2, sharedFeatures], axis=-1)
    dense = Dense(500, activation='sigmoid')(concatenateLayer)
    softmax = Dense(numberOfClasses, activation='softmax')(dense)

    model = Model(inputs=[sentence1, sentence2,  sharedFeatures], outputs=[softmax])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if (singlePrint):
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=150, verbose=0)
        print(history.history["acc"])
    else :
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=150)

    plot_Training(history)

    print("-----TRAINING COMPLETE-----")
    return model

def train_NN_classifier(dataset, epochs, singlePrint=False):

    print("-----TRAIN CLASSIFIER-----")

    x_train, y_train = dl.prepare_data_for_NN(dataset)
    numberOfClasses = y_train.shape[1]

    model = Sequential()
    model.add(Dense(500, input_dim=len(x_train[0]), activation='sigmoid'))
    model.add(Dense(numberOfClasses, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if (singlePrint):
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=150, verbose=0)
        print(history.history["acc"])
    else :
        history = model.fit(x_train, y_train,  validation_split=0.2, epochs=epochs, batch_size=150)

    plot_Training(history)

    print("-----TRAINING COMPLETE-----")
    return model

def train_RF_classifier(dataset):

    print("-----TRAIN CLASSIFIER-----")

    x_train, y_train = dl.prepare_data_for_RF(dataset)

    estimators = 200

    randomForest = RandomForestClassifier(n_estimators=estimators)

    randomForest.fit(x_train, y_train)

    print("-----TRAINING COMPLETE-----")
    return randomForest

def train_Dummy_classifier(train):

    x_train, y_train = dl.prepare_data_for_ZeroR(train)

    classifier = DummyClassifier(strategy="stratified",random_state=0)
    classifier.fit(x_train, y_train)

    return classifier

def plot_Training(history):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def test_classifier(dataset, classifier):

    print("-----TESTING CLASSIFIER-----")

    if isinstance(classifier, keras.engine.training.Model):

        x_test, y_test = dl.prepare_data_for_RNN(dataset)

        print("-----TEST SET SIZE: "+str(x_test["sentence1"].shape)+"-----")
        scores = classifier.evaluate(x_test, y_test)
        print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))

    elif isinstance(classifier, keras.models.Sequential):

        x_test, y_test = dl.prepare_data_for_NN(dataset)

        print("-----TEST SET SIZE: "+str(len(x_test))+"-----")
        scores = classifier.evaluate(x_test, y_test)
        print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))

    elif isinstance(classifier, sklearn.ensemble.forest.RandomForestClassifier):

        x_test, y_test = dl.prepare_data_for_RF(dataset)
        print("-----TEST SET SIZE: "+str(len(x_test))+"-----")

    else:
        x_test, y_test = dl.prepare_data_for_ZeroR(dataset)
        print("-----TEST SET SIZE: "+str(len(x_test))+"-----")

    prediction = classifier.predict(x_test)

    numberOfClasses = y_test.shape[1]

    position = np.argmax(prediction, axis=-1)
    y_pred = np.identity(numberOfClasses)[position]

    target_names = ['nonrelated', 'related']
    print(classification_report(y_test, y_pred, target_names=target_names))

    y_test = [np.where(r==1)[0][0] for r in y_test]
    y_pred = [np.where(r==1)[0][0] for r in y_pred]

    y_true = pd.Series(y_test)
    y_pred = pd.Series(y_pred)

    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

def baseline_results(dataset):

    x_test, y_test = dl.prepare_data_for_ZeroR(dataset)

    prediction = np.random.choice(2, x_test.shape[0])
    numberOfLabels = np.unique(prediction).shape[0]
    prediction = np.identity(numberOfLabels)[prediction.astype(int).flatten()]

    numberOfClasses = y_test.shape[1]

    position = np.argmax(prediction, axis=-1)
    y_pred = np.identity(numberOfClasses)[position]

    target_names = ['nonrelated', 'related']
    print(classification_report(y_test, y_pred, target_names=target_names))

    y_test = [np.where(r==1)[0][0] for r in y_test]
    y_pred = [np.where(r==1)[0][0] for r in y_pred]

    y_true = pd.Series(y_test)
    y_pred = pd.Series(y_pred)

    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


def change_labels(dataset):

    dataset = dataset[(dataset.label != 2) & (dataset.label != -2)]
    dataset.loc[dataset.label == -1, 'label'] = 1

    return dataset

