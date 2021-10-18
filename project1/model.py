from torch import nn
import torch


class BpNet(nn.Module):
    def __init__(self, net_size):
        super(BpNet, self).__init__()
        # this is a two-layer fully-connected network
        self.net = nn.Sequential(nn.Linear(net_size[0], net_size[1]), \
                                 nn.ReLU(), \
                                 nn.Linear(net_size[1], net_size[2]), \
                                 nn.ReLU(), \
                                 nn.Linear(net_size[2], net_size[3]))

    def forward(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt)


class RbfNet(nn.Module):
    def __init__(self, centers, num_class):
        super(RbfNet, self).__init__()
        self.num_class = num_class
        self.num_centers = centers.size(0)
        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1, self.num_centers) / 10)
        self.linear = nn.Linear(self.num_centers, self.num_class, bias=True)
        self.initialize_weights()

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input,
                         -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A -
                                      B).pow(2).sum(2, keepdim=False).sqrt()))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(radial_val)
        return class_score

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


class GRNNet(nn.Module):
    def __init__(self, x_training, y_training):
        super(GRNNet, self).__init__()
        self.x_training = x_training
        self.y_training = y_training
        self.beta = nn.Parameter(torch.ones(1, self.x_training.size(0)) / 0.1)

    def kernel_fun(self, batches):
        A = self.x_training.repeat(batches.size(0), 1, 1)
        B = batches.unsqueeze(1).repeat(1, self.x_training.size(0), 1)
        C = torch.exp(-self.beta.mul((A -
                                      B).pow(2).sum(2, keepdim=False).sqrt()))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        psum1 = radial_val.sum(axis=1).view(-1, 1).repeat(
            1, self.y_training.size(1))  #view(-1,1)
        psum2 = torch.mm(radial_val, self.y_training)
        output = psum2 / psum1
        return output
