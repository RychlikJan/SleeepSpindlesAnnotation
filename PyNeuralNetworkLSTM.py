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
import sys
def saveToCSV(dataX, dataY, name):
    #data = np.concatenate((dataX,dataY))
    #np.savetxt("data.csv", dataX, delimiter=";")


    with open("DELETEresultTest" + str(name)+".csv", "ab") as f:
        np.savetxt(f, dataX, delimiter=";")
    with open("DELETEresultPredict" + str(name)+".csv", "ab") as f:
        np.savetxt(f, dataY, delimiter=";")


def main():
    if(len(sys.argv) != 5):
        print("Wrong arguments, expect 5")
        exit()
    dim = 32
    max = 0.00039350522621499963
    min = -0.00039420439459830623
    dataframe = pandas.read_csv(sys.argv[1], header=None, delimiter=";")
    #dataframe = pandas.read_csv("source32nonSpinRedForTrain.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    trainX = dataset[:, 0:dim]
    dataframe = pandas.read_csv(sys.argv[2], header=None, delimiter=";")
    #dataframe = pandas.read_csv("output32nonSpinRedForTrain.csv", header=None, delimiter=";")
    dataset = dataframe.values
    trainY = dataset[:, 0:dim]
    trainX = (trainX -min) / (max - min)

    print("Printing train data")
    print(trainX)
    print(trainX.shape)
    print(trainY.shape)

    # load validation csv file
    dataframe = pandas.read_csv(sys.argv[3], header=None, delimiter=";")
    #dataframe = pandas.read_csv("source32nonSpinRedToTest.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    testX = dataset[:, 0:dim]
    dataframe = pandas.read_csv(sys.argv[4], header=None, delimiter=";")
    #dataframe = pandas.read_csv("output32nonSpinRedToTest.csv", header=None, delimiter=";")
    dataset = dataframe.values
    testY = dataset[:, 0:dim]

    print("Printing validation data")
    testX = (testX - min) / (max - min)
    print(testX)
    print(testX.shape)
    print(testY.shape)

    # create model
    model = Sequential()
    print("started")
    # TODO hidden layer
    model.add(Embedding(32268, 1, input_length=dim))
    #model.add(Embedding(top_words, 64, input_length=64))
    model.add(LSTM(150,activation='relu'))
    model.add(Dense(200, activation='sigmoid'))


    #model.add(LSTM(16, activation='sigmoid'))
    # TODO output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # prints model summary
    print("Fitting")
    model.summary()

    # Fit the model


    # evaluate the model
    print("Evaluating")
    for i in range(1):
        model.fit(trainX, trainY, epochs=10, batch_size=1024)
        print("-----------------------------------------------------------------------------------")
        scores = model.evaluate(testX, testY)
        print("-----------------------------------------------------------------------------------")

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
    model.fit(trainX, trainY, epochs=10, batch_size=1024)

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