# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-04 13:41

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from torch import optim

input_dim = 5
data_points = 100

# 首先使用sklearn的二进制类创建一些合成数据
X, y = make_classification(data_points, input_dim, n_informative=3, random_state=101)
X = X.astype(np.float32)
y = y.astype(np.float32)

comb_list = list(combinations([v for v in range(5)], 2))

# 分别展示X1、X2、X3、X4之间的图表关系
fig, ax = plt.subplots(5, 2, figsize=(10, 18))
axes = ax.ravel()
for i, c in enumerate(comb_list):
    j, k = c
    axes[i].scatter(X[:, j], X[:, k], c=y, edgecolor='k', s=200)
    axes[i].set_xlabel("X"+str(j), fontsize=15)
    axes[i].set_ylabel("X"+str(k), fontsize=15)
plt.show()
# 从上述结果中可以看出，这些数据集无法通过简单的线性分类器进行分离
# 神经网络是解决此问题的合适的机器学习工具

# 设置网络结构参数
# 输入层尺寸必须与输入X的尺寸匹配。输出尺寸仅为1，这是一个简单的二进制分类问题。
n_input = X.shape[1]
# 第一个隐藏层神经元个数
n_hidden1 = 8
# 第二个隐藏层神经元个数
n_hidden2 = 4
# 输出神经元个数
n_output = 1


class MyNetwork(nn.Module):
    """
    神经网络类，继承自nn.Module
    """
    def __init__(self):
        """
        初始化网络
        定义了两个隐藏层（这里均为线性运算），对隐藏层的输出进行ReLU激活
        """
        super(MyNetwork, self).__init__()
        # 第一个隐藏层
        self.hidden1 = nn.Linear(n_input, n_hidden1)
        # 激活函数ReLU函数
        self.relu1 = nn.ReLU()
        # 第二个隐藏层
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        # 激活函数ReLU函数
        self.relu2 = nn.ReLU()
        # 输出层
        self.output = nn.Linear(n_hidden2, n_output)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        """
        前向传播
        :param x:
        :param kwargs:
        :return:
        """
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


# 需要将数据转化为tensor
X = torch.from_numpy(X)
y = torch.from_numpy(y)
# 实例化一个模型对象
model = MyNetwork()
# 打印该对象可以看到网络结构
print(model)

# 选择合适的损失函数
# 这里我们选择二进制交叉熵损失
criterion = nn.BCELoss()

# 通过上述定义的神经网络模型运行输入数据集，即一次向前通过并计算输出概率
# 由于权重已初始化为随机，因此将看到随机输出概率（大多数接近0.5）
# 注意，该网络目前尚未训练
result = model.forward(X)
# 打印出前10个概率
print("First 10 probabilities...\n", result[:10])

# 打印所有输出，可以看到输出概率基本都接近0.5
result_numpy = result.detach().numpy().flatten()
plt.figure(figsize=(15, 3))
plt.title("Output probabilities with the untrained model", fontsize=16)
plt.bar([i for i in range(100)], height=result_numpy)
plt.show()

# 在计算了神经网络模型的输出之后，就可以简单地将这些预测值与真实值一起传递给损失函数，以计算总损失
# 在训练期间，将一遍又一遍地进行此计算
loss = criterion(result, y)
print(loss.item())

"""
PyTorch具有Autograd的出色功能（即自动微分），该功能可以跟踪并计算神经网络中使用的所有张量的梯度（导数）。 
在某些情况下，我们可以关闭此功能，但是默认情况下，当张量流过网络时，梯度信息会保留（.require_grad = True）。 
这样，可以使用链规则计算后向梯度，这是反向传播算法的核心。
但是，要计算梯度，我们需要执行tensor.backward()方法。 
下面显示，当我们尝试打印第二个隐藏层权重的梯度时，得到None。但是在执行了loss.backward()之后，我们可以正确地计算出梯度。
"""
print("before computing gradient:\n", model.hidden2.weight.grad)
loss.backward()
print("after computing gradient:\n", model.hidden2.weight.grad)

# 选择合适优化器
# 优化器的目的是使得损失loss最小
# 这里选择学习率为0.1的SGD
optimizer = optim.SGD(model.parameters(), lr=0.1)

"""
# 一次训练过程
# 重置梯度，不进行累积传递
optimizer.zero_grad()
# 前向传播
output = model.forward(X)
# 计算损失
loss = criterion(output, y)
# 反向传播
loss.backward()
# 优化更新参数
optimizer.step()
"""

# 训练模型
epochs = 1000  # 训练次数
running_loss = []
for i in range(epochs):
    # 重置梯度，不进行累积传递
    optimizer.zero_grad()
    # 前向传播
    output = model.forward(X)
    # 计算损失
    loss = criterion(output, y)
    print(f"Epoch - {i + 1}, Loss - {round(loss.item(), 3)}")
    # 反向传播
    loss.backward()
    # 优化更新参数
    optimizer.step()
    # 将每次训练后的loss保存
    running_loss.append(loss.item())
    # 每50次训练后查看变化情况
    # 显然未经训练的网络输出接近，即在正类别和负类别之间没有区别
    # 随着训练的继续，概率彼此分离，通过调整网络的权重逐渐尝试匹配真实的概率分布
    if i != 0 and (i + 1) % 50 == 0:
        logits = model.forward(X).detach().numpy().flatten()
        plt.figure(figsize=(15, 3))
        plt.title("Output probabilities after {} epochs".format(i + 1))
        plt.bar([i for i in range(100)], height=logits)
        plt.show()

# 绘制损失函数变化曲线
# 可以看到随着训练次数增加，损失逐渐减小
plt.figure(figsize=(7, 4))
plt.title("Loss over epochs", fontsize=16)
plt.plot([e for e in range(epochs)], running_loss)
plt.grid(True)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Training loss", fontsize=15)
plt.show()
