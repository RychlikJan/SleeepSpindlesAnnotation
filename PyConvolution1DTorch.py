import pandas
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset


if __name__ == "__main__":

    dim = 568
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

    x_min = trainX.min()
    x_max = trainX.max()

    trainX = (trainX - x_min) / (0.5 * (x_max - x_min)) - 1.0
    x = trainX
    y = trainY[::, 0]

    dataframe = pandas.read_csv("source64nonSpinRedToTest.csv", header=None, delimiter=";")
    print(dataframe.head())
    dataset = dataframe.values
    testX = dataset[:, 0:dim]
    dataframe = pandas.read_csv("output64nonSpinRedToTest.csv", header=None, delimiter=";")
    dataset = dataframe.values
    testY = dataset[:, 0:dim]

    testX = (testX - x_min) / (0.5 * (x_max - x_min)) - 1.0
    x_val = testX
    y_val = testY[::, 0]

    model_layers = []
    model_layers.append(torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3))
    model_layers.append(torch.nn.ReLU())
    model_layers.append(torch.nn.Dropout(p=0.15))
    model_layers.append(torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3))
    model_layers.append(torch.nn.ReLU())
    model_layers.append(torch.nn.Dropout(p=0.2))

    model_layers.append(torch.nn.MaxPool1d(kernel_size=3, stride=2))

    model_layers.append(torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3))
    model_layers.append(torch.nn.ReLU())
    model_layers.append(torch.nn.Dropout(p=0.15))
    model_layers.append(torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3))
    model_layers.append(torch.nn.ReLU())
    model_layers.append(torch.nn.Dropout(p=0.2))

    model_layers.append(torch.nn.MaxPool1d(kernel_size=3, stride=2))

    model_layers.append(torch.nn.Flatten())
    model_layers.append(torch.nn.Linear(in_features=384, out_features=64)) # 768
    model_layers.append(torch.nn.ReLU())
    model_layers.append(torch.nn.Linear(in_features=64, out_features=64))
    model_layers.append(torch.nn.ReLU())
    model_layers.append(torch.nn.Linear(in_features=64, out_features=1))
    #model_layers.append(torch.nn.Sigmoid())

    model = torch.nn.Sequential(*model_layers).cuda()
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    x_tensor = torch.Tensor(x.reshape([x.shape[0], 1, x.shape[1]]))
    y_tensor = torch.Tensor(y)
    dataset = TensorDataset(x_tensor, y_tensor)
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
            pred = pred[::, 0]
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
            for id_batch, (x_batch, y_batch) in enumerate(data_loader_val):
                print(f"Batch: {id_batch}", end="\r")

                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

                with torch.no_grad():
                    pred = model(x_batch)[::, 0]
                    loss = bce2(pred, y_batch)
                    epoch_loss += loss.cpu() * x_batch.size()[0]
                    trues += ((sig(pred) > 0.5) == (y_batch > 0.5)).sum().detach().cpu().numpy()

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

    plt.figure()
    plt.plot(loss_val_list)
    plt.title("Loss on validation dataset")
    plt.show()

    plt.figure()
    plt.plot(acc_val_list)
    plt.title("Accuracy on validation dataset")
    plt.show()

