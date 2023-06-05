import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as sm
import time
import glob
import copy

##数据集
fileset = glob.glob("artdata/traindata*")

trainX_, trainY_ = [], []
for file in fileset:
    traindata = np.loadtxt(file, dtype=float, unpack=True)
    train_ag, train_disp = traindata[0], traindata[1]
    trainX_.append(train_ag)
    trainY_.append(train_disp)
trainX_, trainY_ = np.array(trainX_), np.array(trainY_)

delta = np.max(trainX_)-np.min(trainX_)
trainX = (trainX_ - np.ones_like(trainX_)*np.min(trainX_))/delta
delta2 = np.max(trainY_)-np.min(trainY_)
trainY = (trainY_ - np.ones_like(trainY_)*np.min(trainY_))/delta2
print(delta)
print(delta2)

testdata = np.loadtxt("artdata/testdata1.txt", dtype=float, unpack=True)
test_ag, test_disp = testdata[0], testdata[1]
testX = test_ag
testX = (testX - np.ones_like(testX)*np.min(trainX_))/delta
testY = test_disp
testY = (testY - np.ones_like(testY)*np.min(trainY_))/delta2


##模型参数
epochs = 20000
input_size = 1
hidden_size = 30
output_size = 1
num_layers = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_steps = len(test_ag)

##构造LSTM模型
class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM_Regression, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        out, _ = self.lstm1(input, None)
        output = self.fc(out)
        return output

if __name__ == "__main__":
    ##训练模型
    model = LSTM_Regression(input_size, hidden_size, output_size, num_layers)
    model = model.to(device)
    loss_function = nn.MSELoss()
    loss_function = loss_function.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.1)
    best_loss = 100
    best_model = None
    best_epoch = 0
    start = time.time()
    print("Training for %d epochs..." % epochs)
    loss_set = []
    for epoch in range(epochs):
        trainx, trainy = trainX, trainY
        trainx = trainx.reshape(-1, num_steps, input_size)
        trainx = torch.from_numpy(trainx).to(torch.float32).to(device)
        trainy = trainy.reshape(-1, num_steps, output_size)
        trainy = torch.from_numpy(trainy).to(torch.float32).to(device)
        pred_trainy = model(trainx)
        loss = loss_function(pred_trainy, trainy)
        loss_set.append(loss.item())
        if loss <= best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1)%100 == 0:
            print('epoch:', epoch+1)
            print('total_loss:', loss.item())
    end = time.time()
    print('Running time: %s S'%(end-start))
    print("epoch:", best_epoch, "best_loss:", best_loss)
    torch.save(best_model.state_dict(), 'LSTM_r0.02north3.params')

    with open("tran_loss2.txt", "w") as f:
        for loss in loss_set:
            f.writelines("%f\n" %loss)

    ##评估模型
    model = model.eval()
    testx, testy = testX, testY
    testx = testx.reshape(-1, num_steps, input_size)
    testx = torch.from_numpy(testx).to(torch.float32).to(device)
    testy = testy.reshape(-1, 1)
    pred_testy = best_model(testx)
    pred_testy = pred_testy.detach().cpu().numpy()
    pred_testy = pred_testy.reshape(-1, 1)
    print("r2_score:", sm.r2_score(testy, pred_testy))
    # pred_testy = pred_testy * delta2 + np.ones_like(pred_testy) * np.min(trainY)
    # testy = testy * delta2 + np.ones_like(testy) * np.min(trainY)
    plt.plot(pred_testy, label='pred')
    plt.plot(testy, label='true')
    plt.legend()
    plt.show()