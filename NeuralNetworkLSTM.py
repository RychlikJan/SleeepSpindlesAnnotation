import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
from numpy import array
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

def saveToCSV(dataX, dataY, name):
    #data = np.concatenate((dataX,dataY))
    #np.savetxt("data.csv", dataX, delimiter=";")
    for i in dataX:
        if(i>0.5):
            print(i)
    for i in dataY:
        if(i>0.5):
            print(i)

    with open("DELETEresultTest" + str(name)+".csv", "ab") as f:
        np.savetxt(f, dataX, delimiter=";")
    with open("DELETEresultPredict" + str(name)+".csv", "ab") as f:
        np.savetxt(f, dataY, delimiter=";")

def main():
    dim = 64
    dataframe = pandas.read_csv("source64nonSpinRedForTrain.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    trainX = dataset[:, 0:dim]
    dataframe = pandas.read_csv("output64nonSpinRedForTrain.csv", header=None, delimiter=";")
    dataset = dataframe.values
    trainY = dataset[:, 0:dim]

    print("Printing train data")
    print(trainX.shape)
    print(trainY.shape)

    # load validation csv file
    dataframe = pandas.read_csv("source64nonSpinRedToTest.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    testX = dataset[:, 0:dim]
    dataframe = pandas.read_csv("output64nonSpinRedToTest.csv", header=None, delimiter=";")
    dataset = dataframe.values
    testY = dataset[:, 0:dim]

    print("Printing validation data")
    print(testX.shape)
    print(testY.shape)

    # create model
    model = Sequential()
    print("started")
    # TODO hidden layer
    model.add(Embedding(50000, 64, input_length=1))
    #model.add(Embedding(top_words, 64, input_length=64))
    model.add(LSTM(64, activation='sigmoid'))
    #model.add(LSTM(64, input_shape=(64, 1), activation='sigmoid'))

    #model.add(LSTM(16, activation='sigmoid'))
    # TODO output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # prints model summary
    print("Fitting")
    model.summary()

    # Fit the model
    model.fit(trainX, trainY, epochs=5, batch_size=1)

    # evaluate the model
    print("Evaluating")
    scores = model.evaluate(testX, testY)

    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("Test number: %d" % (1))
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    saveToCSV(trainPredict,testPredict,1)


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
             break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        return array(X), array(y)


def main2():


    # define input sequence
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    # demonstrate prediction
    x_input = array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)


if __name__ == "__main__":
    main()
    #main2()