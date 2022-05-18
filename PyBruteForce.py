import pandas
from matplotlib import pyplot as plt
import sys

def modifyPred(currRes):
    pom = currRes.data
    for i in range(pom.shape[0]):
        if pom[i][0]>pom[i][1]:
            pom[i] = [1.0,0.0]
        else:
            pom[i] = [0.0, 1.0]
    return pom


def compare(currRes, y_batch):
    same = 0;
    diff = 0;
    for i in range(currRes.shape[0]):
        if((currRes[i][0] == y_batch[i][0]) and (currRes[i][1] == y_batch[i][1])):
            same = same+1
        else:
            diff = diff+1
    print("-------------------------------------")
    print("Same = " + str(same))
    print("Diff = " + str(diff))
    print("-------------------------------------")
    return [same, diff]


if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print("Wrong arguments, expect 3")
        exit()
    dim = 568
    dataframe = pandas.read_csv(sys.argv[1], header=None, delimiter=";")
    #dataframe = pandas.read_csv("source64nonSpinRedConvolutionalForTrain.csv", header=None, delimiter=";")
    dataset = dataframe.values
    trainX = dataset[:, 0:dim]
    dataframe = pandas.read_csv(sys.argv[2], header=None, delimiter=";")
    #dataframe = pandas.read_csv("output64nonSpinRedConvolutionalForTrain.csv", header=None, delimiter=";")
    dataset = dataframe.values
    trainY = dataset[:, 0:dim]


    x_min = trainX.min()
    x_max = trainX.max()

    trainX = (trainX - x_min) / (0.5 * (x_max - x_min)) - 1.0
    samePrev = 0
    same = 0
    diff = 0
    counter =0
    allSames = []
    allDiff = []
    border = 24.0
    step = 0.01
    maxSame =0
    maxBorder = 0
    while 1:
        counter = counter +1
        same = 0
        diff = 0
        for i in range(trainX.shape[0]):
            value = 0

            for j in range(trainX.shape[1]):
                value = value + abs(trainX[i][j])

            if(value < border):
                # no spindle
                spindle = [0.0,1.0]
            else:
                #is spindle
                spindle = [1.0,0.0]

            if ((spindle[0] == trainY[i][0]) and (spindle[1] == trainY[i][1])):

                same = same + 1
            else:
                diff = diff + 1
        if(same > maxSame):
            maxSame = same
            maxBorder = border
        allSames.append(same)
        allDiff.append(diff)
        border = border+step
        # if(samePrev <= same):
        #     samePrev = same
        # else:
        #     print("---------------------")
        #     print("Same " + str(same))
        #     print("Diff " + str(diff))
        #     print("Border " + str(border))
        #     print("---------------------")
        #     plt.figure()
        #     plt.plot(allSames)
        #     plt.title("Same in time")
        #     plt.show()
        #
        #     plt.figure()
        #     plt.plot(allDiff)
        #     plt.title("Diff in time")
        #     plt.show()
        #    break
        if(counter % 10 == 0):
            print("---------------------")
            print("Same " + str(same))
            print("Diff " + str(diff))
            print("Border " + str(border))
            print("Max same" + str(maxSame))
            print("maxBorder" + str(maxBorder))
            print("---------------------")

        if(counter % 100 == 0):
            plt.figure()
            plt.plot(allSames)
            plt.title("Same in time")
            plt.show()

            plt.figure()
            plt.plot(allDiff)
            plt.title("Diff in time")
            plt.show()
        if(border > 150):
            break
