import sys


def get_network(netname, num_classes=1000, use_gpu=True):
    """ return given network
    """

    if netname[:3] == 'vgg':
        from models.vgg import vgg
        net = vgg(netname, num_classes)
    elif netname == 'googlenet':
        from models.googlenet import GoogLeNet
        net = GoogLeNet(num_classes)
    elif netname == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes)
    elif netname == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes)
    elif netname == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes)
    elif netname == 'resnext50_32x4d':
        from models.resnet import resnext50_32x4d
        net = resnext50_32x4d(num_classes)
    elif netname == 'resnext101_32x8d':
        from models.resnet import resnext101_32x8d
        net = resnext101_32x8d(num_classes)

    else:
        print('unknown network, killed')
        sys.exit()

    if use_gpu:
        net = net.cuda()

    return net