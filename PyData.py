import random

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
withSpindlesGlobal =0
withoutSpindlesGlobal = 0
skippedSpindlesGlobal = 0
preSpindleGlobal = 0
postSpindleGlobal =0
lenOfSpindle = 0
totalLenOfspindle = 0


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


def setMaxMin(duration, peaks):
    global min
    global max
    global lenOfSpindle
    global totalLenOfspindle
    if(duration> max):
        max = duration
    if(duration< min):
        min = duration
    totalLenOfspindle  = totalLenOfspindle + peaks;
    if(peaks > lenOfSpindle):
        lenOfSpindle = peaks


def printStatistics():
    global min
    global max
    global lenOfSpindle
    global fullCount
    global totalLenOfspindle
    print("Informatios about data: ")
    print("Max size of spindle:         " + str(max))
    print("Min size of spindle:         " + str(min))
    print("Length size of spindle:      " + str(lenOfSpindle))
    print("Average count of spindles:   " + str(totalLenOfspindle/fullCount))
    print("Total count of spindles      " + str(fullCount))


def saveToCSV(dataX, dataY, name):
    #data = np.concatenate((dataX,dataY))
    #np.savetxt("data.csv", dataX, delimiter=";")
    with open("source64nonSpinRed" + str(name)+".csv", "ab") as f:
        np.savetxt(f, dataX, delimiter=";")
    with open("output64nonSpinRed" + str(name)+".csv", "ab") as f:
        np.savetxt(f, dataY, delimiter=";")


def countOfPeaks(oneset, duration, times, i, start):
    peaks = 0;
    currPos = start
    while  currPos < len(times):
        if (times[currPos] > oneset[i]):
            peaks = peaks+1
        if(times[currPos] > (oneset[i] + duration[i])):
            return peaks,currPos
        currPos = currPos+1


def dataStatistics(data, annotations):
    if (debug == 1):
        print("Data statistic")
    global fullCount
    duration = annotations.duration
    oneset = annotations.onset
    #8 channel is also interesting
    fullCount += len(oneset)
    pos = 0
    for i in range(len(oneset)):
        peaks, pos = countOfPeaks(oneset, duration, data.times, i, pos)
        setMaxMin(duration[i],peaks)

def findNearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def initCovolutionalData(data, annotations, filePos):
    if (debug == 1):
        print("Initing data for convolution nerual network: " + str(filePos))
    global fullCount
    duration = annotations.duration
    oneset = annotations.onset
    fullCount += len(oneset)
    raw_data = data.get_data()
    dataX = []
    dataY = []
    EEGchannel = 17 #channel position
    print(data.info.ch_names[EEGchannel])
    print("Count of spindles: " + str(len(duration)))
    withoutSpindle = 0
    withSpindle = 0
    skipped = 0
    arrayLen = len(oneset)
    print("Array Len =" + str(arrayLen))
    step = 0.00390625
    sizeMaxSpin = 284
    #TODO The time to work with data
    for i in range(len(oneset)):
        middleOfspindleReal = round((oneset[i] + (duration[i]/2))/step)
        layerInputSpindle = raw_data[EEGchannel][middleOfspindleReal-sizeMaxSpin:middleOfspindleReal + sizeMaxSpin]
        dataX.append(layerInputSpindle)
        dataY.append([1])  # true
        withSpindle += 1
        middleOfspindleReal = middleOfspindleReal + 2*sizeMaxSpin
        layerInputNonSpindle = raw_data[EEGchannel][middleOfspindleReal - sizeMaxSpin:middleOfspindleReal + sizeMaxSpin]
        dataX.append(layerInputNonSpindle)
        dataY.append([0])  # False
        withoutSpindle += 1
        if (debug == 1):
            if (i % 100000 == 0):
                print("worked= " + str((i * 100) / arrayLen) + "%")

    if(filePos >= 15):
        saveToCSV(dataX,dataY,"ConvolutionalToTest")
    else:
        saveToCSV(dataX, dataY, "ConvolutionalForTrain")
    print("With " + str(withSpindle))
    print("without " + str(withoutSpindle))
    print("Skipped "+ str(skipped))

    global withSpindlesGlobal
    withSpindlesGlobal += withSpindle
    global withoutSpindlesGlobal
    withoutSpindlesGlobal += withoutSpindle
    global skippedSpindlesGlobal
    skippedSpindlesGlobal +=skipped



def initLSTMData(data, annotations, filePos):
    if (debug == 1):
        print("Initing data for nerual network: " + str(filePos))
    global fullCount
    duration = annotations.duration
    oneset = annotations.onset
    fullCount += len(oneset)
    raw_data = data.get_data()
    dataX = []
    dataY = []
    EEGchannel = 17 #channel position
    print(data.info.ch_names[EEGchannel])
    print("Count of spindles: " + str(len(duration)))
    sleepSpindlePos= 0;
    pos = 0
    withoutSpindle = 0
    withSpindle = 0
    skipped = 0
    arrayLen = raw_data.shape[1]
    print("Array Len =" + str(arrayLen))
    sleepSpindleBorder1 = oneset[sleepSpindlePos]
    sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
    #TODO The time to work with data
    while pos <= (arrayLen-(FIRST_LAYER_SIZE)):
        layerInput = raw_data[EEGchannel][pos:pos+FIRST_LAYER_SIZE]
        timeInput = data.times[pos:pos+FIRST_LAYER_SIZE]
        if(timeInput[len(timeInput)-1] <sleepSpindleBorder1):
            #layerInput = np.append(layerInput)
            if(random.randint(0,100)==50):
                dataX.append(layerInput)
                dataY.append([0]) # False
                withoutSpindle += 1
        elif(timeInput[0] > sleepSpindleBorder1 and sleepSpindleBorder2> timeInput[len(timeInput)-1]):
            #layerInput = np.append(layerInput)
            dataX.append(layerInput)
            dataY.append([1]) # true
            withSpindle += 1
        else:
            skipped += 1

        #learnNetwork(layerInput, containsSpindle)

        if(timeInput[0] > sleepSpindleBorder2):
            sleepSpindlePos +=1
            if(sleepSpindlePos >= len(duration)):
                sleepSpindlePos = len(duration)-1
                break
            sleepSpindleBorder1 = oneset[sleepSpindlePos]
            sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
        if(debug==1):
            if(pos%100000 == 0):
                print("worked= " + str((pos * 100)/arrayLen) + "%")
        pos += FIRST_LAYER_SIZE

    if(filePos >= 15):
        saveToCSV(dataX,dataY,"ToTest")
    else:
        saveToCSV(dataX, dataY, "ForTrain")
    print("With " + str(withSpindle))
    print("without " + str(withoutSpindle))
    print("Skipped "+ str(skipped))

    global withSpindlesGlobal
    withSpindlesGlobal += withSpindle
    global withoutSpindlesGlobal
    withoutSpindlesGlobal += withoutSpindle
    global skippedSpindlesGlobal
    skippedSpindlesGlobal +=skipped




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


def initLSTM4ClasesData(data, annotations, filePos):
    if (debug == 1):
        print("Initing data for nerual network 4 Clases: " + str(filePos))
    global fullCount
    noSpindle, preSpindle,spindle, postSpindle = [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]
    preSpindleCount, postSpindleCount = 0,0
    duration = annotations.duration
    oneset = annotations.onset
    fullCount += len(oneset)
    raw_data = data.get_data()
    dataX = []
    dataY = []
    EEGchannel = 17  # channel position
    print(data.info.ch_names[EEGchannel])
    print("Count of spindles: " + str(len(duration)))
    sleepSpindlePos = 0;
    pos = 0
    withoutSpindle = 0
    withSpindle = 0
    skipped = 0
    arrayLen = raw_data.shape[1]
    print("Array Len =" + str(arrayLen))
    sleepSpindleBorder1 = oneset[sleepSpindlePos]
    sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
    # TODO The time to work with data
    while pos <= (arrayLen - (FIRST_LAYER_SIZE)):
        layerInput = raw_data[EEGchannel][pos:pos + FIRST_LAYER_SIZE]
        timeInput = data.times[pos:pos + FIRST_LAYER_SIZE]
        if (timeInput[len(timeInput) - 1] < sleepSpindleBorder1):
            if (random.randint(0, 100) == 50):
                dataX.append(layerInput)
                dataY.append(noSpindle)  # False
                withoutSpindle += 1
        elif(timeInput[0] < sleepSpindleBorder1 and sleepSpindleBorder1 <timeInput[len(timeInput) - 1]):
            dataX.append(layerInput)
            dataY.append(preSpindle)
            preSpindleCount +=1
        elif (timeInput[0] > sleepSpindleBorder1 and sleepSpindleBorder2 > timeInput[len(timeInput) - 1]):
            dataX.append(layerInput)
            dataY.append(spindle)  # true
            withSpindle += 1
        elif(timeInput[0] < sleepSpindleBorder2 and sleepSpindleBorder2 <timeInput[len(timeInput) - 1]):
            dataX.append(layerInput)
            dataY.append(postSpindle)
            postSpindleCount += 1
            sleepSpindlePos += 1
            if (sleepSpindlePos >= len(duration)):
                sleepSpindlePos = len(duration) - 1
                break
            sleepSpindleBorder1 = oneset[sleepSpindlePos]
            sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
        else:
            skipped += 1

        # learnNetwork(layerInput, containsSpindle)

        if (timeInput[0] > sleepSpindleBorder2):
            sleepSpindlePos += 1
            if (sleepSpindlePos >= len(duration)):
                sleepSpindlePos = len(duration) - 1
                break
            sleepSpindleBorder1 = oneset[sleepSpindlePos]
            sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
        if (debug == 1):
            if (pos % 100000 == 0):
                print("worked= " + str((pos * 100) / arrayLen) + "%")
        pos += FIRST_LAYER_SIZE

    if (filePos >= 15):
        saveToCSV(dataX, dataY, "4ClassesToTest")
    else:
        saveToCSV(dataX, dataY, "4ClassesForTrain")
    print("With " + str(withSpindle))
    print("without " + str(withoutSpindle))
    print("Pre spindle " + str(preSpindleCount))
    print("Pre spindle " + str(postSpindleCount))
    print("Skipped " + str(skipped))

    global withSpindlesGlobal
    withSpindlesGlobal += withSpindle
    global withoutSpindlesGlobal
    withoutSpindlesGlobal += withoutSpindle
    global skippedSpindlesGlobal
    skippedSpindlesGlobal += skipped
    global preSpindleGlobal
    preSpindleGlobal += preSpindleCount
    global postSpindleGlobal
    postSpindleGlobal += postSpindleCount


def initConvolutionalData(data, annotations, filePos):
    if (debug == 1):
        print("Initing data for nerual network 4 Clases: " + str(filePos))
    global fullCount
    noSpindle, preSpindle,spindle, postSpindle = [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]
    preSpindleCount, postSpindleCount = 0,0
    duration = annotations.duration
    oneset = annotations.onset
    fullCount += len(oneset)
    raw_data = data.get_data()
    dataX = []
    dataY = []
    EEGchannel = 17  # channel position
    print(data.info.ch_names[EEGchannel])
    print("Count of spindles: " + str(len(duration)))
    sleepSpindlePos = 0;
    pos = 0
    withoutSpindle = 0
    withSpindle = 0
    skipped = 0
    arrayLen = raw_data.shape[1]
    print("Array Len =" + str(arrayLen))
    sleepSpindleBorder1 = oneset[sleepSpindlePos]
    sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
    # TODO The time to work with data
    while pos <= (arrayLen - (FIRST_LAYER_SIZE)):
        layerInput = raw_data[EEGchannel][pos:pos + FIRST_LAYER_SIZE]
        timeInput = data.times[pos:pos + FIRST_LAYER_SIZE]
        if (timeInput[len(timeInput) - 1] < sleepSpindleBorder1):
            if (random.randint(0, 100) == 50):
                dataX.append(layerInput)
                dataY.append(noSpindle)  # False
                withoutSpindle += 1
        elif(timeInput[0] < sleepSpindleBorder1 and sleepSpindleBorder1 <timeInput[len(timeInput) - 1]):
            dataX.append(layerInput)
            dataY.append(preSpindle)
            preSpindleCount +=1
        elif (timeInput[0] > sleepSpindleBorder1 and sleepSpindleBorder2 > timeInput[len(timeInput) - 1]):
            dataX.append(layerInput)
            dataY.append(spindle)  # true
            withSpindle += 1
        elif(timeInput[0] < sleepSpindleBorder2 and sleepSpindleBorder2 <timeInput[len(timeInput) - 1]):
            dataX.append(layerInput)
            dataY.append(postSpindle)
            postSpindleCount += 1
            sleepSpindlePos += 1
            if (sleepSpindlePos >= len(duration)):
                sleepSpindlePos = len(duration) - 1
                break
            sleepSpindleBorder1 = oneset[sleepSpindlePos]
            sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
        else:
            skipped += 1

        # learnNetwork(layerInput, containsSpindle)

        if (timeInput[0] > sleepSpindleBorder2):
            sleepSpindlePos += 1
            if (sleepSpindlePos >= len(duration)):
                sleepSpindlePos = len(duration) - 1
                break
            sleepSpindleBorder1 = oneset[sleepSpindlePos]
            sleepSpindleBorder2 = sleepSpindleBorder1 + duration[sleepSpindlePos]
        if (debug == 1):
            if (pos % 100000 == 0):
                print("worked= " + str((pos * 100) / arrayLen) + "%")
        pos += FIRST_LAYER_SIZE

    if (filePos >= 15):
        saveToCSV(dataX, dataY, "4ClassesToTest")
    else:
        saveToCSV(dataX, dataY, "4ClassesForTrain")
    print("With " + str(withSpindle))
    print("without " + str(withoutSpindle))
    print("Pre spindle " + str(preSpindleCount))
    print("Pre spindle " + str(postSpindleCount))
    print("Skipped " + str(skipped))

    global withSpindlesGlobal
    withSpindlesGlobal += withSpindle
    global withoutSpindlesGlobal
    withoutSpindlesGlobal += withoutSpindle
    global skippedSpindlesGlobal
    skippedSpindlesGlobal += skipped
    global preSpindleGlobal
    preSpindleGlobal += preSpindleCount
    global postSpindleGlobal
    postSpindleGlobal += postSpindleCount



def main():
    print("Hello World!")
    now = datetime.now()
    startTime = now.strftime("%H:%M:%S")
    print("Start Time =", startTime)
    for i in range(19):
        names = [0 for x in range(2)]
        buildName(i+1, names)
        annotations = getAnnotations(names[1])
        data = getData(names[0])

        #dataStatistics(data,annotations)
        #printSpindles(data,annotations)

        #initLSTMData(data, annotations, i+1)
        #initLSTM4ClasesData(data, annotations, i + 1)
        #initConvolutionalData(data, annotations, i + 1)
        initCovolutionalData(data,annotations,i+1)
        #printStatistics()

    print(fullCount)
    print(min)
    print(max)
    now2 = datetime.now()
    end = now2.strftime("%H:%M:%S")
    print("End time = ", end)
    print("With " + str(withSpindlesGlobal))
    print("without " + str(withoutSpindlesGlobal))
    print("Skipped " + str(skippedSpindlesGlobal))
    print("Pre spindle "+ str(preSpindleGlobal))
    print("Post spindle " + str(postSpindleGlobal))
    # vykresleni dat od urcite vzdalenosti nekam


if __name__ == "__main__":
    main()