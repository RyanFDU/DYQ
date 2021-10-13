#coding:utf8
'''
This is a small network to do a regression work in training.txt and test.txt
Four network class were put into model.py
Remember the net_size param controls the choice of target y as well.
You can freely predict all last 5 params in 1 net by using MyDataset3
'''
import numpy as np
from sklearn.preprocessing import scale
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from model import BpNet, RbfNet
import matplotlib.pyplot as plt


class MyDataset1(Dataset):  # predict G_f
    def __init__(self, file) -> None:
        self.data = np.loadtxt(file, dtype=np.float32)
        # standardization by column
        self.x = scale(self.data[:, :5])
        self.y = scale(self.data[:, 5])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.data.shape[0]


class MyDataset2(Dataset):  # predict last 4 params
    def __init__(self, file) -> None:
        self.data = np.loadtxt(file, dtype=np.float32)
        # standardization by column
        self.x = scale(self.data[:, :5])
        self.y = scale(self.data[:, 6:])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.data.shape[0]


class MyDataset3(Dataset):  # predict all 5 params in one model
    def __init__(self, file) -> None:
        self.data = np.loadtxt(file, dtype=np.float32)
        # standardization by column
        self.x = scale(self.data[:, :5])
        self.y = scale(self.data[:, 5:])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.data.shape[0]


def run_net(n_epochs, learning_rate, net_size, network_type, plot_loss=False):
    # Check if GPU is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load data w.r.t net_size
    if net_size[-1] == 1:
        dataset1 = MyDataset1(file='./project1/training.txt')
        dataset2 = MyDataset1(file='./project1/test.txt')
    elif net_size[-1] == 4:
        dataset1 = MyDataset2(file='./project1/training.txt')
        dataset2 = MyDataset2(file='./project1/test.txt')
    elif net_size[-1] == 5:
        dataset1 = MyDataset3(file='./project1/training.txt')
        dataset2 = MyDataset3(file='./project1/test.txt')
    else:
        print('output wrong size!')
        return
    training_set = DataLoader(dataset1, dataset1.__len__(), shuffle=True)
    test_set = DataLoader(dataset2, dataset2.__len__(), shuffle=False)

    # Define network, loss function, optimizer
    if network_type == 'BpNet':
        net = BpNet(net_size).to(device)
    if network_type == 'RbfNet':
        centers = torch.rand(net_size[0], net_size[1])
        net = RbfNet(centers, net_size[2]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    avg_tr_loss, avg_tt_loss = [], []
    for epoch in range(n_epochs):
        # Training
        net.train()
        total_tr_loss = 0
        for x, y in training_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = net(x)
            tr_loss = criterion(pred, y)
            tr_loss.backward()
            optimizer.step()
            total_tr_loss += tr_loss.item()

        # Validation
        net.eval()
        total_tt_loss = 0
        for x, y in test_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():  # disable gradient calculation
                pred = net(x)
                tt_loss = criterion(pred, y)
            total_tt_loss += tt_loss.item() * len(x)

        print('epoch: {} test loss: {:.4f}'.format(epoch, tt_loss))
        avg_tr_loss.append(total_tr_loss / len(test_set.dataset))
        avg_tt_loss.append(total_tt_loss / len(test_set.dataset))

    if plot_loss:
        fig = plt.figure(dpi=150, figsize=(8, 4))
        f1 = fig.add_subplot(121)
        f1.plot(np.arange(len(avg_tr_loss)), avg_tr_loss)
        f1.set_title('training loss')
        f2 = fig.add_subplot(122)
        f2.plot(np.arange(len(avg_tt_loss)), avg_tt_loss)
        f2.set_title('test loss')
        plt.show()


if __name__ == '__main__':
    # run_net(1000, 0.01, (5, 5, 12, 4), 'BpNet', plot_loss=True)   # bp_net
    run_net(1000, 0.01, (300, 5, 4), 'RbfNet', plot_loss=True)  # rbf_net
