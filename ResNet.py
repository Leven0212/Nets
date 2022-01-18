# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: ResNet.py
# Date: 2022/1/11 0011
import torch
import torch.nn as nn


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, s):
        super(Conv_block, self).__init__()
        o1, o2, o3 = out_channels
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, o1, 1, stride=s, padding=0),
            nn.BatchNorm2d(o1),
            nn.ReLU(),
            nn.Conv2d(o1, o2, 3, stride=1, padding=1),
            nn.BatchNorm2d(o2),
            nn.ReLU(),
            nn.Conv2d(o2, o3, 1, stride=1, padding=0),
            nn.BatchNorm2d(o3)
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, o3, 1, stride=s, padding=0),
            nn.BatchNorm2d(o3)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x_short = self.stage1(x)
        x = self.stage0(x)
        x = x + x_short
        return self.relu(x)


class Indentity_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Indentity_block, self).__init__()
        o1, o2, o3 = out_channels
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels, o1, 1, stride=1, padding=0),
            nn.BatchNorm2d(o1),
            nn.ReLU(),
            nn.Conv2d(o1, o2, 3, stride=1, padding=1),
            nn.BatchNorm2d(o2),
            nn.ReLU(),
            nn.Conv2d(o2, o3, 1, stride=1, padding=0),
            nn.BatchNorm2d(o3)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x_short = x
        x = self.stage(x)
        x = x + x_short
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, n_class):
        super(ResNet, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.stage1 = nn.Sequential(
            Conv_block(64, [64, 64, 256], 1),
            Indentity_block(256, [64, 64, 256]),
            Indentity_block(256, [64, 64, 256])
        )
        self.stage2 = nn.Sequential(
            Conv_block(256, [128, 128, 512], 2),
            Indentity_block(512, [128, 128, 512]),
            Indentity_block(512, [128, 128, 512]),
            Indentity_block(512, [128, 128, 512])
        )
        self.stage3 = nn.Sequential(
            Conv_block(512, [256, 256, 1024], 2),
            Indentity_block(1024, [256, 256, 1024]),
            Indentity_block(1024, [256, 256, 1024]),
            Indentity_block(1024, [256, 256, 1024]),
            Indentity_block(1024, [256, 256, 1024]),
            Indentity_block(1024, [256, 256, 1024])
        )
        self.stage4 = nn.Sequential(
            Conv_block(1024, [512, 512, 2048], 2),
            Indentity_block(2048, [512, 512, 2048]),
            Indentity_block(2048, [512, 512, 2048])
        )
        self.pool = nn.AvgPool2d(2, 2, padding=1)
        self.linear = nn.Linear(2048 * 4 * 4, n_class)

    def forward(self, x):
        b = x.shape[0]
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.reshape(b, -1)
        return self.linear(x)


if __name__ == '__main__':
    x = torch.rand((15, 3, 224, 224))
    print(x.shape)
    net = ResNet(1000)
    out = net.forward(x)
    print(out.shape)
