import mne
import pyedflib
import numpy as np
from datetime import datetime


debug = 1
path = "D:\\ZCU\\5.rocnik\\Diplomová práce\\data\\SS2\\"
fullCount = 0
max = 0
min = 1000000
FIRST_LAYER_SIZE = 64
SECOND_LAYER_SIZE = 32
THIRD_LAYER_SIZE = 16
OUTPUT_LAYER_SIZE = 0

def getAnnotations(file):
    if(debug == 1):
        print("Get annotations from file: " + file)
    data = mne.io.read_raw_edf(file)
    annotations = data._annotations
    return annotations


def getData(file):
    if (debug == 1):
        print("Get data from file: " + file)
    data = mne.io.read_raw_edf(file)
    return data


def setMaxMin(duration):
    global min
    global max
    if(duration> max):
        max = duration
    if(duration< min):
        min = duration


def learnNetwork(layerInput, containsSpindle):
    print(layerInput)


def saveToCSV(dataX, dataY):
    #data = np.concatenate((dataX,dataY))
    #np.savetxt("data.csv", dataX, delimiter=";")
    with open("data.csv", "ab") as f:
        np.savetxt(f, dataX, delimiter=";")


def initData(data, annotations, filePos):
    if (debug == 1):
        print("Initing data for nerual network: " + str(filePos))
    global fullCount
    duration = annotations.duration
    oneset = annotations.onset
    chan_idxs = [data.ch_names.index(ch) for ch in data.ch_names]
    fullCount += len(oneset)
    raw_data = data.get_data()
    dataX = []
    dataY = []
    EEGchannel = 17
    print(data.info.ch_names[EEGchannel])
    print("Count of spindles: " + str(len(duration)))
    sleepSpindlePos= 0;
    pos = 0
    withoutSpindle = 0
    withSpindle = 0
    skipped = 0
    arrayLen = raw_data.shape[1]
    arrayLen = 300
    print("Array Len =" + str(arrayLen))
    sleepSpindleBorder1 = oneset[sleepSpindlePos]
    sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
    #TODO The time to work with data
    while pos <= (arrayLen-(FIRST_LAYER_SIZE)):
        layerInput = raw_data[EEGchannel][pos:pos+FIRST_LAYER_SIZE]
        timeInput = data.times[pos:pos+FIRST_LAYER_SIZE]
        if(timeInput[len(timeInput)-1] <sleepSpindleBorder1):
            layerInput = np.append(layerInput,0)
            dataX.append(layerInput)
            #dataY.append([0]) # False
            withoutSpindle += 1
        elif(timeInput[0] > sleepSpindleBorder1 and sleepSpindleBorder2< timeInput[len(timeInput)-1]):
            layerInput = np.append(layerInput,1)
            dataX.append(layerInput)
            #dataY.append([1]) # true
            withSpindle += 1
        else:
            skipped += 1

        #learnNetwork(layerInput, containsSpindle)

        if(timeInput[0] > sleepSpindleBorder2):
            sleepSpindlePos +=1
            if(sleepSpindlePos >= len(duration)):
                sleepSpindlePos = len(duration)-1

            sleepSpindleBorder1 = oneset[sleepSpindlePos]
            sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
        if(debug==1):
            if(pos%100000 == 0):
                print("worked= " + str((pos * 100)/arrayLen) + "%")
        pos += FIRST_LAYER_SIZE
    
    saveToCSV(dataX,dataY)
    print("With " + str(withSpindle))
    print("without " + str(withoutSpindle))
    print("Skipped "+ str(skipped))



def printSpindles(data, annotations):
    if (debug == 1):
        print("Printing data")
    global fullCount
    duration = annotations.duration
    oneset = annotations.onset
    #8 channel is also interesting
    print(data.info.ch_names[17])
    chan_idxs = [data.ch_names.index(ch) for ch in data.ch_names]
    fullCount += len(oneset)
    for i in range(len(oneset)):
        setMaxMin(duration[i])
        data.plot(order=chan_idxs, start=oneset[i], duration=duration[i]+2)


def buildName(pos, names):
    if(pos < 10 ):
        names[0] = path + "data\\01-02-000"+str(pos)+" PSG.edf"
        names[1] = path + "Annotations\\01-02-000"+str(pos)+" Spindles_E1.edf"
    else:
        names[0] = path + "data\\01-02-00" + str(pos) + " PSG.edf"
        names[1] = path + "Annotations\\01-02-00" + str(pos) + " Spindles_E1.edf"
    if (debug == 1):
        print("Names generated: " + names[0] + ", " + names[1])


def main():
    print("Hello World!")
    now = datetime.now()
    startTime = now.strftime("%H:%M:%S")
    print("Start Time =", startTime)
    for i in range(1):
        names = [0 for x in range(2)]
        buildName(i+1, names)
        annotations = getAnnotations(names[1])
        data = getData(names[0])
        #printSpindles(data,annotations)
        initData(data, annotations, i+1)
    print(fullCount)
    print(min)
    print(max)
    now2 = datetime.now()
    end = now.strftime("%H:%M:%S")
    print("End time = ", end)
    length = now2-now;
    fullTime = length.strftime("%H:%M:%S")
    print("Full Time =",fullTime)
    # vykresleni dat od urcite vzdalenosti nekam


if __name__ == "__main__":
    main()