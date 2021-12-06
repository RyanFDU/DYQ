import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import utils as vutils
from tensorboardX import SummaryWriter

import utils
from utils import Cifar10Loader, CAM
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# An implement of transfer-learning from Imagenet to Cifar10
# Take resnet34 as an example


def train(net_name, image_path, split_ratio, batch_size, epochs, lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Set multi-threading params
    n_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0,
                     3])  # number of workers
    print('Using {} dataloader workers every process'.format(n_workers))

    # Data pre-processing
    cifar_sets = Cifar10Loader(image_path)
    train_dataset, validate_dataset = cifar_sets.get_train_val_set(split_ratio)
    test_dataset = cifar_sets.get_test_set()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=n_workers)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=n_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=n_workers)

    # Initial the network
    net = utils.get_network(net_name, use_gpu=True) if torch.cuda.is_available() \
        else utils.get_network(net_name, use_gpu=False)

    # Check if pre-trained weights exists
    net_weight_path = "./params/resnet34-pre.pth"
    if not os.access(net_weight_path, os.W_OK):
        synset_url = 'https://download.pytorch.org/nets/resnet34-333f7ec4.pth'
        os.system('wget ' + synset_url)
        os.system('mkdir ./params')
        os.system('mv ./resnet34-333f7ec4.pth ./params/resnet34-pre.pth')
    net.load_state_dict(torch.load(net_weight_path, map_location=device))

    # Gradient fixed (accelerate training)
    for param in net.parameters():
        param.requires_grad = False

    # Change the size of fully-connected layer
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, cifar_sets.num_classes)
    nn.init.xavier_uniform_(net.fc.weight)
    net.to(device)

    # Define loss function
    loss_function = nn.CrossEntropyLoss()

    # Construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr)

    # Start training
    best_acc = 0.0
    save_path = './params/resNet34.pth'
    train_steps = len(train_loader)
    avg_tr_loss, avg_tr_acc, avg_val_acc = [], [], []
    for epoch in range(epochs):
        # train
        net.train()
        acc_tr = 0.0
        training_loss = 0.0
        train_bar = tqdm(train_loader)
        for _, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            predict_y_tr = torch.max(logits, dim=1)[1]

            # print statistics
            training_loss += loss.item()
            acc_tr += torch.eq(predict_y_tr, labels.to(device)).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, epochs, loss)
        tr_accuracy = acc_tr / cifar_sets.train_size

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accuracy = acc / cifar_sets.val_size
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, training_loss / train_steps, val_accuracy))

        avg_tr_loss.append(training_loss / train_steps)
        avg_tr_acc.append(tr_accuracy)
        avg_val_acc.append(val_accuracy)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

    # plot
    plt.figure(figsize=(20, 8), dpi=100)
    plt.title('loss curve and accuracy curves')
    plt.plot(np.arange(len(avg_tr_loss)),
             avg_tr_loss,
             color='green',
             label='training loss')
    plt.plot(np.arange(len(avg_tr_loss)),
             avg_tr_acc,
             color='red',
             label='training accuracy')
    plt.plot(np.arange(len(avg_tr_loss)),
             avg_val_acc,
             color='blue',
             label='testing accuracy')
    plt.legend()
    plt.xlabel('epoches')
    plt.ylabel('loss & accuracy')
    plt.show()

    # test
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        all_preds = torch.tensor([]).to(device)
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            all_preds = torch.cat((all_preds, outputs), dim=0)
            test_bar.desc = "test epoch[{}/{}]".format(1, 1)

    test_accuracy = acc / cifar_sets.test_size
    print(' test_accuracy: %.3f' % (test_accuracy))

    cm = confusion_matrix(test_dataset.targets, all_preds.argmax(dim=1).cpu())
    print(cm)
    names = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
             "horse", "ship", "truck")
    plt.figure(figsize=(10, 10))
    utils.plot_confusion_matrix(cm, names)

    print('Finished Training')


def predict(net_name, img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # image pre-processing
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(
        img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = img_transform(img)
    # expand batch dimension into [N, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(
        json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    num_classes = len(class_indict)

    # Initial the network with trained weights
    net = utils.get_network(net_name, num_classes, use_gpu=True) if torch.cuda.is_available() \
        else utils.get_network(net_name, num_classes, use_gpu=False)
    weights_path = "./params/resNet34.pth"
    net.load_state_dict(torch.load(weights_path, map_location=device))

    # -------------------Start of feature map visualization-------------------
    # access Tensorboard by follow command:
    # tensorboard --logdir=./visulization

    # define Summary_Writer
    writer = SummaryWriter('./visulization')
    img_grid = vutils.make_grid(img, normalize=True, scale_each=True, nrow=2)
    writer.add_image('raw img', img_grid, global_step=666)
    print(img.size())

    net.eval()
    img0 = img
    for name, layer in net._modules.items():
        img0 = img0.view(img0.size(0), -1) if "fc" in name else img0
        print(img0.size())

        img0 = layer(img0.to(device))
        print(f'{name}')

        # the first conv layer have no relu
        img0 = F.relu(img0) if 'conv' in name else img0
        if 'layer' in name or 'conv' in name:
            img1 = img0.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
            img_grid = vutils.make_grid(img1,
                                        normalize=True,
                                        scale_each=True,
                                        nrow=4)
            writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)
    # -------------------End of feature map visualization-------------------

    # ----------------------Start of CAM visualization----------------------
    cam = CAM()
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(net._modules.items()):
        module = cam.recursion_change_bn(net)
    net.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    net.eval()

    # hook the feature extractor
    features_names = ['layer4',
                      'avgpool']  # this is the last conv layer of the resnet
    for name in features_names:
        net._modules.get(name).register_forward_hook(cam.hook_feature)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(net(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    params = list(net.parameters())
    weight_softmax = params[-2].data.cpu().numpy()
    weight_softmax[weight_softmax < 0] = 0

    print('Class activation map is saved as CAM.jpg')
    CAMs = cam.returnCAM(cam.features_blobs[0], weight_softmax, [predict_cla])

    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)),
                                cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img * 0.5
    cv2.imwrite('./CAM.jpg', result)
    result = result.astype(np.uint8)
    cv2.imshow('', result)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    # ----------------------End of CAM visualization----------------------


def test_accuracy(net_name, image_path, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Set multi-threading params
    n_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0,
                     3])  # number of workers
    print('Using {} dataloader workers every process'.format(n_workers))

    # Data pre-processing
    cifar_sets = Cifar10Loader(image_path)
    test_dataset = cifar_sets.get_test_set()
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=n_workers)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(
        json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    num_classes = len(class_indict)

    # Initial the network with trained weights
    net = utils.get_network(net_name, num_classes, use_gpu=True) if torch.cuda.is_available() \
        else utils.get_network(net_name, num_classes, use_gpu=False)
    weights_path = "./params/resNet34.pth"
    net.load_state_dict(torch.load(weights_path, map_location=device))

    # test
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        all_preds = torch.tensor([])
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            all_preds = torch.cat((all_preds, outputs), dim=0)
            test_bar.desc = "test epoch[{}/{}]".format(1, 1)

    test_accuracy = acc / cifar_sets.test_size
    print(' test_accuracy: %.3f' % (test_accuracy))

    cm = confusion_matrix(test_dataset.targets, all_preds.argmax(dim=1))
    print(cm)
    names = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
             "horse", "ship", "truck")
    plt.figure(figsize=(10, 10))
    utils.plot_confusion_matrix(cm, names)
    print('')


if __name__ == '__main__':
    # train('resnet34',
    #       './images',
    #       split_ratio=0.8,
    #       batch_size=8,
    #       epochs=3,
    #       lr=0.0001)
    predict('resnet34', './images/test/17.jpg')

    # for name in os.listdir('./images/test/'):
    #     path = './images/test/'+name
    #     # print(path)
    #     name = name.split('.')[0]
    #     predict('resnet34', 10, path, name)
