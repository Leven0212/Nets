# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: GoogLeNet.py
# Date: 2021/12/22 0022

import torch
import torch.nn as nn


class Inception_block(nn.Module):
    def __init__(self, in_channeles, out_1x1, in_3x3, out_3x3, in_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()

        self.branch1 = conv_block(in_channeles, out_1x1, kernel_size=(1, 1))
        self.branch2 = nn.Sequential(
            conv_block(in_channeles, in_3x3, kernel_size=(1, 1)),
            conv_block(in_3x3, out_3x3, kernel_size=(3, 3), padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channeles, in_5x5, kernel_size=(1, 1)),
            conv_block(in_5x5, out_5x5, kernel_size=(5, 5), padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            conv_block(in_channeles, out_1x1pool, kernel_size=(1, 1))
        )

    def forward(self, x):
        # N x filters x 28 x 28
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class conv_block(nn.Module):
    def __init__(self, in_channles, out_channles, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channles, out_channles, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channles)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3):
        super(GoogLeNet, self).__init__()
        self.conv1 = conv_block(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)

        self.inception3a = Inception_block(in_channeles=192, out_1x1=64, in_3x3=96, out_3x3=128, in_5x5=16, out_5x5=32,
                                           out_1x1pool=32)
        self.inception3b = Inception_block(in_channeles=256, out_1x1=128, in_3x3=128, out_3x3=192, in_5x5=32,
                                           out_5x5=96, out_1x1pool=64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, 2, 1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        self.pool5 = nn.AvgPool2d(7, 1)

        self.dropout = nn.Dropout(.4)
        self.linear = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.pool5(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x
