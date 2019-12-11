# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-05 14:20

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from torchvision import transforms

from MyFacialKeyPointsDataset import MyFacialKeyPointsDataset
from MyTransforms import Rescale, RandomCrop, Normalize, ToTensor

key_pts_frame = pd.read_csv('./data/training_frames_keypoints.csv')

idx = 0
image_name = key_pts_frame.iloc[idx, 0]
key_pts = key_pts_frame.iloc[idx, 1:].values
key_pts = key_pts.astype('float').reshape(-1, 2)

print("Image name: {}".format(image_name))
print("Landmarks shape: {}".format(key_pts.shape))
print("First 4 key pts:\n {}".format(key_pts[:4]))
print("Number of images: {}".format(key_pts_frame.shape[0]))


def show_keypoints(image, keypoints):
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=20, marker='.', c='m')


plt.figure(figsize=(5, 5))
show_keypoints(mpimg.imread(os.path.join('data/training/', image_name)), key_pts)
plt.show()

# 测试转换，以确保它们的行为符合预期
face_dataset = MyFacialKeyPointsDataset(csv_file='./data/training_frames_keypoints.csv', root_dir='./data/training/')
test_num = 500
sample = face_dataset[test_num]
rescale = Rescale(100)
crop = RandomCrop(50)
composed = transforms.Compose([Rescale(250), RandomCrop(224)])
for i, tx in enumerate([rescale, crop, composed]):
    transformed_sample = tx(sample)
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    show_keypoints(transformed_sample['image'], transformed_sample['keypoints'])
plt.show()

# 应用变换以获得相同形状的灰度图像
data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
transformed_dataset = MyFacialKeyPointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                               root_dir='./data/training/',
                                               transform=data_transform)
print("images：{}".format(transformed_dataset))
for i in range(5):
    sample_transform = transformed_dataset[i]
    print("{}, {}, {}".format(i, sample_transform['image'].size(), sample_transform['keypoints'].size()))
