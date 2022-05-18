import pandas
import torch
import sys
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset


def modifyPred(currRes):
    pom = currRes.data
    #print(pom)
    for i in range(pom.shape[0]):
        if pom[i][0]>pom[i][1]:
            pom[i] = torch.Tensor([1.0,0.0])
        else:
            pom[i] = torch.Tensor([0.0, 1.0])
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
    if(len(sys.argv) != 5):
        print("Wrong arguments, expect 5")
        exit()

    dim = 568
    #dataframe = pandas.read_csv("source64nonSpinRedForTrain.csv", header=None, delimiter=";")
    dataframe = pandas.read_csv(sys.argv[1], header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    trainX = dataset[:, 0:dim]
    #dataframe = pandas.read_csv("output64nonSpinRedForTrain.csv", header=None, delimiter=";")
    dataframe = pandas.read_csv(sys.argv[2], header=None, delimiter=";")
    dataset = dataframe.values
    trainY = dataset[:, 0:dim]

    print("Printing train data")
    print(trainX.shape)
    print(trainY.shape)

    xMin = trainX.min()
    xMax = trainX.max()

    trainX = (trainX - xMin) / (0.5 * (xMax - xMin)) - 1.0
    x = trainX
    y = trainY[::, ::]

    #dataframe = pandas.read_csv("source64nonSpinRedToTest.csv", header=None, delimiter=";")
    dataframe = pandas.read_csv(sys.argv[3], header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    testX = dataset[:, 0:dim]
    #dataframe = pandas.read_csv("output64nonSpinRedToTest.csv", header=None, delimiter=";")
    dataframe = pandas.read_csv(sys.argv[4], header=None, delimiter=";")
    dataset = dataframe.values
    testY = dataset[:, 0:dim]

    testX = (testX - xMin) / (0.5 * (xMax - xMin)) - 1.0
    x_val = testX
    y_val = testY[::, ::]

    modelLayers = []
    modelLayers.append(torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3))
    modelLayers.append(torch.nn.ReLU())
    modelLayers.append(torch.nn.Dropout(p=0.15))
    modelLayers.append(torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3))
    modelLayers.append(torch.nn.ReLU())
    modelLayers.append(torch.nn.Dropout(p=0.2))

    modelLayers.append(torch.nn.MaxPool1d(kernel_size=3, stride=2))

    modelLayers.append(torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3))
    modelLayers.append(torch.nn.ReLU())
    modelLayers.append(torch.nn.Dropout(p=0.15))
    modelLayers.append(torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3))
    modelLayers.append(torch.nn.ReLU())
    modelLayers.append(torch.nn.Dropout(p=0.2))

    modelLayers.append(torch.nn.MaxPool1d(kernel_size=3, stride=2))

    modelLayers.append(torch.nn.Flatten())
    modelLayers.append(torch.nn.Linear(in_features=384, out_features=64)) # 768
    modelLayers.append(torch.nn.ReLU())
    modelLayers.append(torch.nn.Linear(in_features=64, out_features=64))
    modelLayers.append(torch.nn.ReLU())
    modelLayers.append(torch.nn.Linear(in_features=64, out_features=2))
    modelLayers.append(torch.nn.Sigmoid())

    model = torch.nn.Sequential(*modelLayers).cuda()
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    xTensor = torch.Tensor(x.reshape([x.shape[0], 1, x.shape[1]]))
    yTensor = torch.Tensor(y)
    dataset = TensorDataset(xTensor, yTensor)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1024, shuffle=True)

    x_val_tensor = torch.Tensor(x_val.reshape([x_val.shape[0], 1, x_val.shape[1]]))
    y_val_tensor = torch.Tensor(y_val)
    dataset_val = TensorDataset(x_val_tensor, y_val_tensor)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=4096, shuffle=False)

    # Train
    best_loss = float("Inf")
    loss_list = []
    loss_val_list = []
    acc_val_list = []
    sames = []
    diff = []
    samesMax = [0,9999999999]

    bce = torch.nn.BCELoss()
    bce2 = torch.nn.BCEWithLogitsLoss()
    sig = torch.nn.Sigmoid()


    epochs = 2500
    for epoch in range(0, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = torch.zeros(1)

        for id_batch, (x_batch, y_batch) in enumerate(data_loader):
            print(f"Batch: {id_batch}", end="\r")

            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            pred = model(x_batch)
            pred = pred[::,::]
            loss = bce2(pred, y_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.cpu() * x_batch.size()[0]

        epoch_loss /= x.shape[0]
        epoch_loss = epoch_loss.detach().numpy()[0]
        print(f"Loss={epoch_loss}")
        loss_list.append(epoch_loss)

        if epoch % 10 == 0:
            epoch_loss = torch.zeros(1)
            trues = 0
            currSame = 0
            currDiff = 0
            for id_batch, (x_batch, y_batch) in enumerate(data_loader_val):
                print(f"Batch: {id_batch}", end="\r")

                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

                with torch.no_grad():
                    pred = model(x_batch)[::, ::]
                    currRes = pred
                    loss = bce2(pred, y_batch)
                    epoch_loss += loss.cpu() * x_batch.size()[0]
                    trues += ((sig(pred) > 0.5) == (y_batch > 0.5)).sum().detach().cpu().numpy()
                    currRes = modifyPred(currRes)
                    samediff = compare(currRes,y_batch)
                    currSame = currSame + samediff[0]
                    currDiff = currDiff + samediff[1]

            if(currSame> samesMax[0]):
                samesMax =[currSame,currDiff]
            sames.append(currSame)
            diff.append(currDiff)

            epoch_loss /= x_val.shape[0]
            epoch_loss = epoch_loss.detach().numpy()[0]
            print(f"Loss on validation dataset={epoch_loss}")
            loss_val_list.append(epoch_loss)
            acc = trues/x_val.shape[0]
            acc_val_list.append(acc)
            print(f"Accuracy on validation dataset={acc}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), "weights.h5")

    model.load_state_dict(torch.load("weights.h5"))

    plt.figure()
    plt.plot(loss_list)
    plt.title("Loss on train dataset")
    plt.show()
    plt.savefig('LossOnTrainDataset.png')

    plt.figure()
    plt.plot(loss_val_list)
    plt.title("Loss on validation dataset")
    plt.show()
    plt.savefig('LossOnValidationDataset.png')

    plt.figure()
    plt.plot(acc_val_list)
    plt.title("Accuracy on validation dataset")
    plt.show()
    plt.savefig('AccurencyOnDataset.png')

    plt.figure()
    plt.plot(sames)
    plt.title("Same values in time- 1D Convolutional")
    plt.show()
    plt.savefig('CountOfSameSamples.png')

    plt.figure()
    plt.plot(diff)
    plt.title("Diff values in time- 1D Convolutional")
    plt.show()
    plt.savefig('CountOfDiffSamples.png')
    print(samesMax[0])
    print(samesMax[1])

