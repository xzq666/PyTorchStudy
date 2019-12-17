# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-16 17:38

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import Net
from MyFacialKeyPointsDataset import MyFacialKeyPointsDataset
from MyTransforms import Rescale, RandomCrop, Normalize, ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn

# 转换
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
# 数据集
transformed_dataset = MyFacialKeyPointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                               root_dir='./data/training',
                                               transform=data_transform)
# 批量加载训练数据
batch_size = 10
train_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
# 创建测试集
test_dataset = MyFacialKeyPointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                        root_dir='./data/test',
                                        transform=data_transform)
# 批量加载测试数据
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)
# 定义网络
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)


def net_sample_output():
    """
    在一个批次测试图像上测试模型
    :return:
    """
    for i, sample in enumerate(test_loader):
        images, key_pts = sample['image'], sample['keypoints']
        images = images.type(torch.FloatTensor)
        # 前向传播
        output_pts = net(images)
        # 将输出reshpe为68个坐标点
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        # 测试一张
        if i == 0:
            return images, output_pts, key_pts


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """
    显示预测的人脸关键点
    :param image:
    :param predicted_key_pts:
    :param test_pts:
    :return:
    """
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=100, marker='.', c='m')
    # 如果有人脸关键点标签 则一起显示
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=100, marker='.', c='g')


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    for i in range(batch_size):
        plt.figure(figsize=(20, 10))
        # 未经转换的图像数据
        image = test_images[i].data
        # 转换为NumPy数组
        image = image.numpy()
        # 转换为NumPy数组格式的图像 HxWxC
        image = np.transpose(image, (1, 2, 0))
        # 未经转换的预测关键点
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # 让预测关键点变为未经转换的数据
        predicted_key_pts = predicted_key_pts * 50.0 + 100
        # 若存在标签 则一起绘制进行比较 需要将其变为未经转换的数据
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
        plt.axis('off')
    plt.show()


# 定义损失函数与优化器
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


def train_net(n_epochs):
    """
    训练网络
    :param n_epochs: 迭代轮数
    :return:
    """
    net.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            images, key_pts = data['image'], data['keypoints']
            # 将关键点摊平（转为一维）
            key_pts = key_pts.view(key_pts.size(0), -1)
            # 将变量转换为浮点数以进行回归损失
            images = images.type(torch.FloatTensor)
            key_pts = key_pts.type(torch.FloatTensor)
            # 重置梯度
            optimizer.zero_grad()
            # 前向传播获取输出
            output_pts = net(images)
            # 获取损失
            loss = criterion(output_pts, key_pts)
            # 反向传播
            loss.backward()
            # 更新
            optimizer.step()
            # 每10次输出一次
            running_loss += loss.item()
            if i % 9 == 0:
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0


# 从小开始 在确定模型结构和超参数时增加
n_epochs = 10
train_net(n_epochs)
# 测试模型
test_images, test_outputs, label_pts = net_sample_output()
visualize_output(test_images, test_outputs, label_pts)
