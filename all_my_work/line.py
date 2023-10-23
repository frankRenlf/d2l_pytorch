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
import numpy as np
import torch.nn as nn
import torch


def pseudo_A(A, b):
    # Ax = b   =>   x = (A.T @ A)^-1 @ A @ b
    if b is None:
        P = A @ np.linalg.inv(A.T @ A) @ A.T
        print(P)
    else:
        x = np.linalg.inv(A.T @ A) @ A.T @ b
        print(x)
        print(A @ x)


def orthogonal():
    A = np.array([[1], [1], [1]], dtype=np.float32)
    b = np.array([[1], [0], [2]])
    # Ax = b   =>   x = (A.T @ A)^-1 @ A @ b
    B = b - A @ (A.T @ b / (A.T @ A))
    print(B)


if __name__ == "__main__":
    pseudo_A(A=np.array([[2], [1], [2]]), b=None)
    # orthogonal()
    # print(np.__version__)
