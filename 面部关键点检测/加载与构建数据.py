# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-05 14:20

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

key_pts_frame = pd.read_csv('./data/training_frames_keypoints.csv')

idx = 0
image_name = key_pts_frame.iloc[idx, 0]
key_pts = key_pts_frame.iloc[idx, 1:].values
key_pts = key_pts.astype('float').reshape(-1, 2)

print("Image name: {}".format(image_name))
print("Landmarks shape: {}".format(key_pts.shape))
print("First 4 key pts:\n {}".format(key_pts[:4]))
print("Number of images: {}".format(key_pts_frame.shape[0]))


def show_keypoints(image, key_pts):
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')


plt.figure(figsize=(5, 5))
show_keypoints(mpimg.imread(os.path.join('data/training/', image_name)), key_pts)
plt.show()
