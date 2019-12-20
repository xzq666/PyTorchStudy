# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-19 19:44

import cv2
import matplotlib.pyplot as plt
import torch
from models import Net
import numpy as np
from torch.autograd import Variable

image = cv2.imread("./images/obamas.jpg", 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 利用已有的训练好的检测器检测人脸
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(image, 1.2, 2)
# image_with_detections = image.copy()
# for (x, y, w, h) in faces:
#     cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3)
# fig = plt.figure(figsize=(9, 9))
# plt.imshow(image_with_detections)


# 利用自己训练的网络进行
# 1 加载训练好的网络
net = Net()
net.load_state_dict(torch.load('./saved_models/krunal_keypoints_model_lr0001_epoch20.pt'))
print(net.eval())


def show_points(image_test, key_points):
    """
    显示检测结果
    :param image_test:
    :param key_points:
    :return:
    """
    plt.figure()
    key_points = key_points.data.numpy()
    key_points = key_points * 60.0 + 68
    key_points = np.reshape(key_points, (68, -1))
    plt.imshow(image_test, cmap='gray')
    plt.scatter(key_points[:, 0], key_points[:, 1], s=50, marker='.', c='r')


image_copy = image.copy()
for (x, y, w, h) in faces:
    # 选择人脸区域
    roi = image_copy[y:y+h, x:x+w]
    # 做灰度处理
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    image_roi = roi
    # image_roi = cv2.resize(image_roi, (224, 224))
    # 标准化灰度图像 使其颜色范围落在[0,1]
    roi = roi / 255.0
    # 将检测到的人脸缩放为训练好的网络的预期正方形尺寸（224x224）
    roi = cv2.resize(roi, (224, 224))
    # 将图像形状（H x W）转换为Tensor所需形状（N x C x H x W）
    roi = np.expand_dims(roi, 0)
    roi = np.expand_dims(roi, 0)
    # 数据预处理
    roi_torch = Variable(torch.from_numpy(roi))
    roi_torch = roi_torch.type(torch.FloatTensor)
    # 面部关键点检测
    keypoints = net(roi_torch)
    show_points(image_roi, keypoints)

plt.show()
