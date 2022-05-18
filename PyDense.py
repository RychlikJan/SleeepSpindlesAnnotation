import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import LSTM
from numpy import zeros, newaxis
from numpy import array
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import sys

def main():
    if(len(sys.argv) != 5):
        print("Wrong arguments, expect 5")
        exit()
    dim = 32
    dataframe = pandas.read_csv(sys.argv[1], header=None, delimiter=";")
    #dataframe = pandas.read_csv("source32nonSpinRedForTrain.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    trainX = dataset[:, 0:dim]
    dataframe = pandas.read_csv(sys.argv[2], header=None, delimiter=";")
    #dataframe = pandas.read_csv("output32nonSpinRedForTrain.csv", header=None, delimiter=";")
    dataset = dataframe.values
    trainY = dataset[:, 0:dim]

    print("Printing train data")
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
    print(testX.shape)
    print(testY.shape)
    max = 0.00039350522621499963
    min = -0.00039420439459830623
    testX = (testX - min) / (max - min)
    trainX = (trainX - min) / (max - min)

    # create model
    model = Sequential()
    print("started")
    # TODO hidden layer
    model.add(Dense(64, input_dim=dim, activation='sigmoid'))

    model.add(Dense(16, activation='sigmoid'))
    # TODO output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # prints model summary
    print("Fitting")
    model.summary()

    # Fit the model
    model.fit(trainX, trainY, epochs=1, batch_size=5)

    # evaluate the model
    print("Evaluating")
    scores = model.evaluate(testX, testY)

    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("Test number: %d" % (1))

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



if __name__ == "__main__":
    main()
    #ma