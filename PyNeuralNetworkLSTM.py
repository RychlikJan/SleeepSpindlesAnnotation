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


    with open("DELETEresultTest" + str(name)+".csv", "ab") as f:
        np.savetxt(f, dataX, delimiter=";")
    with open("DELETEresultPredict" + str(name)+".csv", "ab") as f:
        np.savetxt(f, dataY, delimiter=";")


def main():
    dim = 568
    dataframe = pandas.read_csv("source64nonSpinRedConvolutionalForTrain.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    trainX = dataset[:, 0:dim]
    dataframe = pandas.read_csv("output64nonSpinRedConvolutionalForTrain.csv", header=None, delimiter=";")
    dataset = dataframe.values
    trainY = dataset[:, 0:dim]

    print("Printing train data")
    print(trainX.shape)
    print(trainY.shape)

    # load validation csv file
    dataframe = pandas.read_csv("source64nonSpinRedConvolutionalToTest.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    testX = dataset[:, 0:dim]
    dataframe = pandas.read_csv("output64nonSpinRedConvolutionalToTest.csv", header=None, delimiter=";")
    dataset = dataframe.values
    testY = dataset[:, 0:dim]

    print("Printing validation data")
    print(testX.shape)
    print(testY.shape)

    # create model
    model = Sequential()
    print("started")
    # TODO hidden layer
    model.add(Embedding(32268, 1, input_length=dim))
    #model.add(Embedding(top_words, 64, input_length=64))
    model.add(LSTM(64,activation='relu'))

    #model.add(LSTM(16, activation='sigmoid'))
    # TODO output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # prints model summary
    print("Fitting")
    model.summary()

    # Fit the model
    model.fit(trainX, trainY, epochs=1, batch_size=1)

    # evaluate the model
    print("Evaluating")
    scores = model.evaluate(testX, testY)

    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("Test number: %d" % (1))
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    print(testPredict)
    saveToCSV(trainPredict,testPredict,1)


def main2():
    dim = 64
    dataframe = pandas.read_csv("source64nonSpinRed4ClassesForTrain.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    trainX = dataset[:, 0:dim]
    dataframe = pandas.read_csv("output64nonSpinRed4ClassesForTrain.csv", header=None, delimiter=";")
    dataset = dataframe.values
    trainY = dataset[:, 0:dim]

    print("Printing train data")
    print(trainX.shape)
    print(trainY.shape)

    # load validation csv file
    dataframe = pandas.read_csv("source64nonSpinRed4ClassesToTest.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    testX = dataset[:, 0:dim]
    dataframe = pandas.read_csv("output64nonSpinRed4ClassesToTest.csv", header=None, delimiter=";")
    dataset = dataframe.values
    testY = dataset[:, 0:dim]

    print("Printing validation data")
    print(testX.shape)
    print(testY.shape)

    # create model
    model = Sequential()
    print("started")
    # TODO hidden layer
    model.add(Embedding(32268,1,input_length=64))
    # model.add(Embedding(top_words, 64, input_length=64))
    model.add(LSTM(64,activation='relu'))

    #model.add(LSTM(16, activation='sigmoid'))
    # TODO output layer
    model.add(Dense(4,activation='relu'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # prints model summary
    print("Fitting")
    model.summary()

    # Fit the model
    model.fit(trainX, trainY, epochs=1, batch_size=1)

    # evaluate the model
    print("Evaluating")
    scores = model.evaluate(testX, testY)

    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("Test number: %d" % (1))
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    print(testPredict)
    saveToCSV(trainPredict, testPredict, 1)



if __name__ == "__main__":
    main()
    #main2()