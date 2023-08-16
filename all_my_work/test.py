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

if __name__ == "__main__":
    tensor = torch.ones(2, 2, 2, 3)
    dd2 = tensor.mean(dim=(0, 2, 3), keepdim=True)
    print(tensor, '\n', dd2)
