import os, sys
import json, itertools
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets


class Cifar10Loader():
    def __init__(self, image_path):
        # Image pre-processing
        self.__normalize = {
            "train":
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            "test":
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }

        # Download datasets
        self.__train_val_set = datasets.CIFAR10(
            root=os.path.join(image_path, "train"),
            train=True,
            download=True,
            transform=self.__normalize["train"])
        self.train_val_size = len(self.__train_val_set)

        self.__test_set = datasets.CIFAR10(root=os.path.join(
            image_path, "val"),
                                           train=False,
                                           download=True,
                                           transform=self.__normalize["test"])
        self.test_size = len(self.__test_set)

        # Get labels
        class_list = self.__train_val_set.class_to_idx
        cla_dict = dict((val, key) for key, val in class_list.items())
        self.num_classes = len(cla_dict)
        ''' cla_dict:
        {"0": "airplane", "1": "automobile", "2": "bird", "3": "cat", "4": "deer",
        "5": "dog", "6": "frog", "7": "horse", "8": "ship", "9": "truck"}
        '''

        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

    def get_train_val_set(self, split_ratio=0.8):
        self.train_size = int(split_ratio * self.train_val_size)
        self.val_size = int(self.train_val_size - self.train_size)
        training_set, validation_set = torch.utils.data.random_split(
            self.__train_val_set, [self.train_size, self.val_size])
        print("using {} images for training, {} images for validation".format(
            self.train_size, self.val_size))
        return training_set, validation_set

    def get_test_set(self):
        print("using {} images for test".format(self.test_size))
        return self.__test_set


class CAM:
    def __init__(self) -> None:
        self.features_blobs = []

    def recursion_change_bn(self, module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = 1
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = self.recursion_change_bn(module1)
        return module

    def hook_feature(self, module, input, output):
        self.features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[class_idx].dot(
                feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam


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


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('./confusion_matrix.png')