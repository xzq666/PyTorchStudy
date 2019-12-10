# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-10 10:40

from torch.utils.data import Dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


class MyFacialKeyPointsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        数据集初始化
        :param csv_file: csv文件路径
        :param root_dir: 所有图像的目录
        :param transform: 要应用的transform
        """
        super(MyFacialKeyPointsDataset, self).__init__()
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        """
        根据索引读取图像
        :param idx: 索引
        :return:
        """
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
        image = mpimg.imread(image_name)
        # 若图像具有Alpha颜色通道，则将其消除
        if image.shape[2] == 4:
            image = image[:, :, :3]
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    """
    用于测试
    """
    face_dataset = MyFacialKeyPointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                            root_dir='./data/training/')
    print("数据集大小：{}".format(len(face_dataset)))
    # 显示一些图像看看有没有出错
    num_to_display = 3
    for i in range(num_to_display):
        fig = plt.figure(figsize=(5, 5))
        random_idx = random.randint(0, len(face_dataset)-1)
        sample_test = face_dataset[random_idx]
        image_test = sample_test['image']
        keypoint_test = sample_test['keypoints']
        plt.imshow(image_test)
        plt.scatter(keypoint_test[:, 0], keypoint_test[:, 1], s=20, marker='.', c='m')
        plt.show()
