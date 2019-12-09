# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-07 14:27

import torch

u = torch.tensor([1, 0])
v = torch.tensor([0, 1])
print(u)
print(v)
# 矢量加法
w = u + v
print(w)
# 标量向量乘法
y = 1.5 * w
print(y)
# 线性组合
a = torch.tensor([1, 2])
b = torch.tensor([3, -1])
print(2 * a + 3 * b)
# 张量乘积
print(a * b)
# 点乘
print(torch.dot(a, b))
# 矩阵乘法
a = a.reshape(2, 1)
b = b.reshape(1, 2)
print(torch.mm(a, b))
# 广播
c = torch.tensor([1, 2, 3, 4])
d = c + 10
print(d)
