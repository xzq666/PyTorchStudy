# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-09 19:14

import torch


def func(input):
    return input**3 - 7*input**2 + 11*input


x = torch.tensor(2.0, requires_grad=True)
y = func(x)
print("x=2处的函数值：{}".format(y))
# 自动求导触发
y.backward()
print("x=2处的grad：{}".format(x.grad))

# 求不同x值处的grad，需要重新定义
x = torch.tensor(3.0, requires_grad=True)
y = func(x)
print("x=3处的函数值：{}".format(y))
y.backward()
print("x=3处的grad：{}".format(x.grad))

# 对于求偏导数也是一样的用法
u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(1.0, requires_grad=True)
f = 3 * u**2 * v + 4 * v**3
f.backward()
print(u.grad)
print(v.grad)

# 对于矢量也会保存一系列grad
a = torch.linspace(-10, 10, requires_grad=True)
b = torch.sum(a**2)
b.backward()
print(a.grad)
