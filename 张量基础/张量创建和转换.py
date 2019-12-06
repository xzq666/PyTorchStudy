# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-06 10:29

import torch
import numpy as np
import pandas as pd

# 一维Tensor创建
a = torch.tensor([i for i in range(5)])
print("a：{}, 类型：{}, 数据类型：{}".format(a, type(a), a.dtype))
print(a.type(torch.float32))
b = torch.FloatTensor([1, 2, 3, 4, 5])
print(b)
# 一维Tensor索引，获取的是Tensor类型
print(a[1])
print(type(a[1]))
# item()方法获取实际值
print(a[1].item())
print(type(a[1].item()))
# size()与ndimension()维度
print(a.size())
print(a.ndimension())
# view()方法改变Tensor视图
a_new1 = a.view(5, 1)
print(a_new1)
a_new2 = a.view(-1, 1)
print(a_new2)
# Tensor与NumPy数组转换
np_array = np.array([1, 3, 5])
print(np_array)
tensor_from_array = torch.from_numpy(np_array)
print(tensor_from_array)
np_array_new = tensor_from_array.numpy()
print(np_array_new)
# 重新转换的numpy数组的指针指向张量对象，该张量对象又指向原始的numpy数组
# 因此，原始numpy数组中的任何更改都会反映回来
np_array *= 2
print(np_array)
print(tensor_from_array)
print(np_array_new)
# Tensor与pandas转换
pd_series = pd.Series([i for i in range(1, 6)])
print(pd_series)
pd_to_torch = torch.from_numpy(pd_series.values)
print(pd_to_torch)
# Tensor转列表
torch_to_list = pd_to_torch.tolist()
print(torch_to_list)
for i in range(len(torch_to_list)):
    torch_to_list[i] *= 2
print(torch_to_list)
print(pd_to_torch)
