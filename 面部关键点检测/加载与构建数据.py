# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-05 14:20

import pandas as pd

key_pts_frame = pd.read_csv('./data/training_frames_keypoints.csv')

idx = 0
image_name = key_pts_frame.iloc[idx, 0]
key_pts = key_pts_frame.iloc[idx, 1:].values
key_pts = key_pts.astype('float').reshape(-1, 2)

print("Image name: {}".format(image_name))
print("Landmarks shape: {}".format(key_pts.shape))
print("First 4 key pts:\n {}".format(key_pts[:4]))

print("Number of images: {}".format(key_pts_frame.shape[0]))
