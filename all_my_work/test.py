# -*- coding: UTF-8 -*-
"""
    @Author : Frank.Ren
    @Project : pytorch 
    @Product : PyCharm
    @createTime : 2023/8/15 15:46 
    @Email : sc19lr@leeds.ac.uk
    @github : https://github.com/frankRenlf
    @Description : 
"""
import torch


def cat_():
    data = [torch.arange(24).reshape(2, 3, 4), torch.arange(24).reshape(2, 3, 4)]
    return torch.cat(data, dim=0).shape


def conv():
    # 创建一个示例数据张量
    X = torch.randn(3, 4, 5, 5)  # 3个样本，4个通道，每个通道的特征图是5x5的

    # 计算通道维度上的均值
    mean = X.mean(dim=(0, 2, 3), keepdim=True)

    # 计算通道维度上的方差
    var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

    print("特征维度上的均值：", mean.shape, mean)
    print("特征维度上的方差：", var.shape, var)


def full():
    # 创建一个示例数据集
    X = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [2.0, 3.0, 4.0, 5.0],
                      [3.0, 4.0, 5.0, 6.0],
                      [4.0, 5.0, 6.0, 7.0],
                      [5.0, 6.0, 7.0, 8.0]])

    # 计算特征维度上的均值
    mean = X.mean(dim=0)

    # 计算特征维度上的方差
    var = ((X - mean) ** 2).mean(dim=0)


if __name__ == "__main__":
    arr = torch.arange(0, 12).reshape(2, 2, -1)
    arr2 = torch.arange(0, 12).reshape(2, 2, -1)
    arr2 = arr2.permute(0, 2, 1)
    print(torch.bmm(arr, arr2))
