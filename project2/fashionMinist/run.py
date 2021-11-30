import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from RunBuilder import RunBuilder, RunManager
from Network import Network
from collections import OrderedDict
from collections import namedtuple
from torch.utils.data import DataLoader

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


# 一、数据准备
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# 使用RunManager和RunBuilder类可以使得程序更易扩展
# 可以直接在这里设置不同的参数，进行多次试验来比较
params = OrderedDict(
    lr=[.01],
    batch_size=[1000, 2000],
    shuffle=[True, False],
    num_workers = [0]
)
m = RunManager()

# 检查版本
print(torch.__version__)
print(torchvision.__version__)

for run in RunBuilder.get_runs(params):
    network = Network()
    loader = DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    m.begin_run(run, network, loader)
    for epoch in range(5):
        m.begin_epoch()
        for batch in loader:
            images, labels = batch
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m.track_loss(loss)
            m.track_num_correct(preds, labels)
        m.end_epoch()
    m.end_run()
m.save('results')
print('End of Train')

# 如果想要查看tensorboard，在Terminal内首先键入“cd project2/fashionMinist”
# 然后键入"tensorboard --logdir=runs"，打开http://localhost:6006/