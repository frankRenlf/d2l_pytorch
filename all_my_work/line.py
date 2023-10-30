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
    test_eig()
    test_eig2()
