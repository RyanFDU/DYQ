import os
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import transforms, datasets

import utils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np


def train(net_name, num_classes, image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Imagenet to Cifar10
    data_transform = {
        "train":
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val":
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = datasets.CIFAR10(root=os.path.join(image_path, "train"), train=True, download=True, transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,
              8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.CIFAR10(root=os.path.join(image_path, "val"), train=False, download=True, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(
        train_num, val_num))

    net = utils.get_network(net_name, use_gpu=False)
    # load pretrain weights
    model_weight_path = "./params/resnet34-pre.pth"
    if not os.access(model_weight_path, os.W_OK):
        synset_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        os.system('wget ' + synset_url)
        os.system('mv ./resnet34-333f7ec4.pth ./params/resnet34-pre.pth')
        
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(
    #     model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, num_classes)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = './params/resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, epochs, loss)

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

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


def predict(netname, num_classes, img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
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
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(
        json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = utils.get_network(netname, num_classes, use_gpu=False)
    weights_path = "./params/resNet34.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    
    model.eval()

    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0

    # logit = model.forward(img)
    # h_x = F.softmax(logit, 1).data.squeeze()
    # probs, idx = h_x.sort(0, True)
    # probs = probs.numpy()
    # idx = idx.numpy()

    print('Class activation map is saved as cam.jpg')
    CAMs = returnCAM(features_blobs[0], weight_softmax, [predict_cla])

    # for i in range(len(predict)):
    #     print('{:.3f} -> {}'.format(probs[i], class_indict[str(idx[i])]))

    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img * 0.5
    cv2.imwrite('cam.jpg', result)
    result = result.astype(np.uint8)
    cv2.imshow('',result)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

if __name__ == '__main__':
    features_blobs = []
    train('resnet34', 10, './images')
    # predict('resnet34', 10, './images/test/3.jpg')