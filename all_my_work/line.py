# -*- coding: UTF-8 -*-
"""
    @Author : Frank.Ren
    @Project : pytorch 
    @Product : PyCharm
    @createTime : 2023/10/17 23:15 
    @Email : e1143935@u.nus.edu
    @github : https://github.com/frankRenlf
    @Description : 
"""
import os

import numpy as np
import torch.nn as nn
import torch
import torchvision
from torchvision.models import ResNet152_Weights, ViT_B_16_Weights


def pseudo_A(A, b, u0):
    # Ax = b   =>   x = (A.T @ A)^-1 @ A @ b
    if b is None:
        # print(A.T @ A)
        P = A @ np.linalg.inv(A.T @ A) @ A.T
        print(P)
        # u1 = P @ u0
        # print(u1)
    else:
        x = np.linalg.inv(A.T @ A) @ A.T @ b
        print(x)
        print(A @ x)


def orthogonal():
    A = np.array([[1], [2], [3]], dtype=np.float32)
    b = np.array([[1], [1], [1]])
    # Ax = b   =>   x = (A.T @ A)^-1 @ A @ b
    B = b - A @ (A.T @ b / (A.T @ A))
    print(B)
    print(b - (6 / 14) * A)


def test_eig():
    A = np.array([[4, 4],
                  [-3, 3]])
    U, D, Vt = np.linalg.svd(A)
    # print("U:", U)
    # print("D:", D)
    print("V:", Vt)


def test_eig2():
    A = np.array([[4, 4],
                  [-3, 3]])
    D, Q = np.linalg.eig(A.T @ A)
    # print('D:', D)
    print("Q:", Q)
    print(1 / np.sqrt(2))


def test_pretrain():
    # @save
    d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                                '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

    # 如果使用Kaggle比赛的完整数据集，请将下面的变量更改为False
    demo = True
    if demo:
        data_dir = d2l.download_extract('dog_tiny')
    else:
        data_dir = os.path.join('..', 'data', 'dog-breed-identification')

    def reorg_dog_data(data_dir, valid_ratio):
        labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
        d2l.reorg_train_valid(data_dir, labels, valid_ratio)
        d2l.reorg_test(data_dir)

    batch_size = 32 if demo else 128
    valid_ratio = 0.1
    reorg_dog_data(data_dir, valid_ratio)

    transform_train = torchvision.transforms.Compose([
        # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
        # 然后，缩放图像以创建224x224的新图像
        torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                 ratio=(3.0 / 4.0, 4.0 / 3.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=15),
        # 随机更改亮度，对比度和饱和度
        torchvision.transforms.ColorJitter(brightness=0.4,
                                           contrast=0.4,
                                           saturation=0.4),
        # 添加随机噪声
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # 从图像中心裁切224x224大小的图片
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])

    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]

    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]

    train_iter, train_valid_iter = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)
        for dataset in (train_ds, train_valid_ds)]

    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                             drop_last=True)

    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                            drop_last=False)
    loss = nn.CrossEntropyLoss(reduction='none')

    def evaluate_loss(data_iter, net, devices):
        l_sum, n = 0.0, 0
        for features, labels in data_iter:
            features, labels = features.to(devices[0]), labels.to(devices[0])
            outputs = net(features)
            l = loss(outputs, labels)
            l_sum += l.sum()
            n += labels.numel()
        return (l_sum / n).to('cpu')

    def get_net(devices):
        finetune_net = nn.Sequential()
        finetune_net.features = torchvision.models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # 定义一个新的输出网络，共有120个输出类别
        finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 120))
        # 将模型参数分配给用于计算的CPU或GPU
        finetune_net = finetune_net.to(devices[0])
        # 冻结参数
        for param in finetune_net.features.parameters():
            param.requires_grad = False
        return finetune_net

    net = get_net([torch.device('cpu')])
    # print(net)
    for i, (features, labels) in enumerate(train_iter):
        features, labels = features, labels
        output = net(features)
        print(output.shape)
        break

    def print_requires_grad(net):
        for name, param in net.named_parameters():
            print(f"{name} requires_grad={param.requires_grad}")

    net = get_net([torch.device('cpu')])
    print(net)


def seq_test():
    data = torch.arange(1000, dtype=torch.float32).reshape(1, 1000)

    def get_net(devices):
        finetune_net = nn.Sequential()
        # 定义一个新的输出网络，共有120个输出类别
        finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 120))
        return finetune_net

    net = get_net([torch.device('cpu')])
    print(net[:-1])


def net_print():
    def get_net(devices):
        finetune_net = nn.Sequential()
        finetune_net.features = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        finetune_net[0].heads = nn.Sequential(nn.Linear(768, 256),
                                              nn.ReLU(), nn.Dropout(0.5),
                                              nn.Linear(256, 120))
        # 将模型参数分配给用于计算的CPU或GPU
        finetune_net = finetune_net.to(devices[0])
        # 冻结参数
        for param in finetune_net.features.parameters():
            param.requires_grad = False
        return finetune_net

    net = get_net([torch.device('cpu')])
    # nn.Dropout()
    print(net)


if __name__ == "__main__":
    # pseudo_A(A=np.array([[1], [2], [3]]), b=np.array([[4], [5], [8]]), u0=np.array([[9], [9], [0]]))
    # pseudo_A(A=np.array([[2], [1], [2]]), b=None, u0=np.array([[9], [9], [0]]))
    # pseudo_A(A=np.array([[0, 1], [1, 0], [0, 2]]), b=None, u0=np.array([[9], [9], [0]]))
    # orthogonal()
    # print(np.__version__)
    # print(19/7)
    from d2l.torch import d2l

    # 2*2*3
    # A = torch.arange(0, 12).reshape(4, 3)
    # b = torch.arange(0, 4).reshape(4)
    # print(d2l.sequence_mask(A, b))
    # test_eig()
    # test_eig2()
    # test_pretrain()
    # seq_test()
    net_print()
