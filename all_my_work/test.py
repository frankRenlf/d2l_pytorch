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
import numpy as np
from d2l import torch as d2l
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


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    # print(X)
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


if __name__ == "__main__":
    batch = 2
    num_heads = 3
    num_steps = 6
    x = torch.arange(0, 72).reshape(batch, num_steps, num_steps)
    dec_valid_lens = torch.arange(
        1, num_steps + 1).repeat(batch, 1)
    valid_lens = dec_valid_lens
    valid_lens = valid_lens.reshape(-1)
    print(valid_lens)
    shape = x.shape
    X = sequence_mask(x.reshape(-1, shape[-1]), valid_lens,
                      value=-1e6)
    print(X)
