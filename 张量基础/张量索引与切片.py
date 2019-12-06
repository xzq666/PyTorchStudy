# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-06 11:08

import torch
import numpy as np

a = torch.Tensor([i for i in range(5)])
print(a)
# 下标索引
print(a[1])
# 修改数据
a[1] = 100
print(a)
# 切片索引
b = a[1:4]
print(b)
# 修改切片数据
a[1:4] = torch.Tensor([-100, -200, -300])
print(a)
