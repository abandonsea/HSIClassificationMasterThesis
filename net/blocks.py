#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:57 2022

@author: Pedro Vieira
@description: Implements residual block to be used in the different 3DCRN architectures
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()

        # Shortcut parameters
        self.batch_norm = nn.BatchNorm3d(input_channels)
        self.relu = nn.ReLU()

        # Block parameters
        self.block = nn.Sequential(nn.Conv3d(input_channels, input_channels, (3, 3, 3), padding=1),
                                   nn.BatchNorm3d(input_channels), self.relu,
                                   nn.Conv3d(input_channels, input_channels, (3, 3, 3), padding=1),
                                   nn.BatchNorm3d(input_channels), self.relu,
                                   nn.Conv3d(input_channels, input_channels, (3, 3, 3), padding=1),
                                   nn.BatchNorm3d(input_channels))

    def forward(self, x):
        out = self.block(x)
        out += self.relu(self.batch_norm(x))

        return out
