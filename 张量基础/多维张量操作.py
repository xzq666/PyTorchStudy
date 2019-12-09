# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-09 18:58

import torch
import numpy as np

list_of_tensor = []
for i in range(3):
    inner_list = []
    for j in range(3):
        inner_list.append(np.random.randint(10, 90))
    list_of_tensor.append(inner_list)
print(list_of_tensor)

list_of_tensor2 = []
for i in range(3):
    inner_list = []
    for j in range(3):
        inner_list.append(np.random.randint(10, 90))
    list_of_tensor2.append(inner_list)
print(list_of_tensor2)

tensor_from_list = torch.tensor(list_of_tensor)
print(tensor_from_list)

# 获取尺寸、形状及大小
print("维度：{}，形状：{}，大小：{}".format(tensor_from_list.ndimension(), tensor_from_list.shape, tensor_from_list.size()))
# 可以通过将size属性转换为ndarray然后应用prod获取元素总数
print("元素总数：{}".format(np.array(tensor_from_list.size()).prod()))
# 张量加法
tensor_from_list2 = torch.tensor(list_of_tensor2)
print(tensor_from_list2)
print(tensor_from_list + tensor_from_list2)
# 标量乘法
print(2.2 * tensor_from_list)
# 索引与切片
print(tensor_from_list[0, 0])
print(tensor_from_list[2, 1:3])
# 张量与张量对应元素相乘
print(tensor_from_list * tensor_from_list2)
# 张量乘法（矩阵乘法）
print(tensor_from_list.mm(tensor_from_list2))
# 转置 dim0, dim1
print(tensor_from_list.transpose(0, 1))
# 逆和行列式
a = torch.tensor([[2.5, 1.2], [3, 2]])
b = torch.tensor([[1.2, 1], [-1, 2.2]])
c = a.mm(b)
print(c)
print(torch.inverse(c))
print(torch.det(c))
