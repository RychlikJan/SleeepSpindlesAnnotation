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
    model.fit(trainX, trainY, epochs=1, batch_size=1)

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