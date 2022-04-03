import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import LSTM
from numpy import zeros, newaxis
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
from numpy import array
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling2D
from keras.layers.convolutional import MaxPooling1D
import tensorflow as tf

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
    trainX = trainX[:,:,newaxis]
    dataframe = pandas.read_csv("output64nonSpinRedConvolutionalForTrain.csv", header=None, delimiter=";")
    dataset = dataframe.values
    trainY = dataset[:, 0:1]

    print("Printing train data")
    print(trainX.shape)
    print(trainY.shape)

    # load validation csv file
    dataframe = pandas.read_csv("source64nonSpinRedConvolutionalToTest.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    testX = dataset[:, 0:dim]

    testX = testX[:,:,newaxis]
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
    model = Sequential()
    model.add(Convolution1D(filters=64, kernel_size=5, activation='relu', input_shape=(568, 1)))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Convolution1D(filters=64, kernel_size=5, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Dense(4, activation='sigmoid'))

    learningO = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam',
    )

    model.compile(loss='binary_crossentropy', optimizer=learningO, metrics=['accuracy'])

    # prints model summary
    print("Fitting")
    print(model.summary())

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
    #main2()