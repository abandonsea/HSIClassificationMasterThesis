#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:57 2022

@author: Pedro Vieira
@description: Implement network architecture of the 3DCRN for the Salinas and Indian Pines datasets
"""

import torch
from torch import nn
from net.blocks import ResidualBlock


class CRN3D(nn.Module):
    def __init__(self, num_classes):
        # Patch size: [1, 20, 23, 23]
        super().__init__()

        self.relu = nn.ReLU()
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 16, (3, 3, 3)),  # Output size: [16, 18, 21, 21]
            nn.BatchNorm3d(16), self.relu,
            ResidualBlock(16), self.relu,
            nn.MaxPool3d((1, 2, 2)),  # Output size: [16, 18, 10, 10]
            nn.Dropout3d(0.05)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3)),  # Output size: [32, 16, 8, 8]
            nn.BatchNorm3d(32), self.relu,
            ResidualBlock(32), self.relu,
            nn.MaxPool3d((1, 2, 2)),  # Output size: [32, 16, 4, 4]
            nn.Dropout3d(0.05),
            nn.Conv3d(32, 64, (3, 3, 3), padding=(1, 0, 0)),  # Output size: [64, 16, 2, 2]
            nn.BatchNorm3d(64), self.relu,
            nn.MaxPool3d((2, 2, 2)),  # Output size: [64, 8, 1, 1]
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), self.relu,
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input size: [batch_size, 1, depth, h, w]
        out = self.block1(x)
        out = self.block2(out)

        out = out.view((-1, 512))
        out = self.classifier(out)
        return out
