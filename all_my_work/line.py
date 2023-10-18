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

if __name__ == "__main__":
    A = np.matrix([[1, 1], [1, 2], [1, 3]])
    b = np.matrix([[1], [2], [2]])
    # Ax = b   =>   x = (A.T @ A)^-1 @ A @ b
    x = np.linalg.inv(A.T @ A) @ A.T @ b
    print(x)
    print(A @ x)
