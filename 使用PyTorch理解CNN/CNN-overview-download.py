# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-13 18:39

from torchvision import datasets
import torchvision.transforms as transforms

num_workers = 0
batch_size = 20

transform = transforms.ToTensor()

datasets.MNIST(root='data', train=True, download=True, transform=transform)
datasets.MNIST(root='data', train=False, download=True, transform=transform)
