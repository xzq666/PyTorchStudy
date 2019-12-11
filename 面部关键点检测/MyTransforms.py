# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-10 19:21

import torch
import cv2
import numpy as np


class Normalize(object):
    """
    将彩色图像转换为灰度并将颜色范围归一化为[0,1]
    """

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        # 转换为灰度图
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        # 将[0, 255]的像素值归一化为[0, 1]
        image_copy = image_copy / 255.0
        # 将关键点定为中心为0，范围为[-1, 1]
        # 均值100，方差为50
        key_pts_copy = (key_pts_copy - 100) / 50.0
        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """
    缩放
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            # 若为整数，则匹配较小的一边，宽高比不变
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        # 若为元组，则匹配宽高
        else:
            new_h, new_w = self.output_size
        # 需要将宽高转换成int型
        new_h, new_w = int(new_h), int(new_w)
        new_image = cv2.resize(image, (new_w, new_h))
        # 将关键点也进行相应缩放
        key_pts = key_pts * [new_w / w, new_h / h]
        return {'image': new_image, 'keypoints': key_pts}


class RandomCrop(object):
    """
    随机裁剪
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        new_image = image[top: top + new_h, left: left + new_w]
        key_pts = key_pts - [left, top]
        return {'image': new_image, 'keypoints': key_pts}


class ToTensor(object):
    """
    转为Tensor张量
    """

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        # 如果图片没有通道，先添加通道
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'keypoints': torch.from_numpy(key_pts)}
