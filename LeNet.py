# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: LeNet.py
# Date: 2021/12/20 0020

import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.relu(self.conv1(x))    # 1*32*32 -> 6*28*28
        x = self.pool(x)                # 6*28*28 -> 6*14*14
        x = self.relu(self.conv2(x))    # 6*14*14 -> 16*10*10
        x = self.pool(x)                # 16*10*10 -> 16*5*5
        x = self.relu(self.conv3(x))    # 16*5*5 -> 120*1*1
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))  # 120*1*1 -> 84*1*1
        x = self.linear2(x)
        return x
