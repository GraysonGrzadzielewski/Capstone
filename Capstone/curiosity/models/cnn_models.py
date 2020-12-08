"""
Module to hold the cnns to embed observations
Only Random for now, maybe more later
"""

import torch
import torch.nn as nn

import numpy as np


class Random(nn.Module):
    """
    Converts an observation of (1, 1, 84, 84) into a feature space of length 512
    """
    def __init__(self, obs_shape, out_size , std=0.0):
        super(Random, self).__init__()
        self.convolution = nn.Sequential(
            # (1, 1, 84, 84) -> channels, color, height, width
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        )
        # Get the output shape of the convolutional network, given an empty input of our data shape
        conv_space = int(np.prod(self.convolution(torch.zeros(1, *obs_shape)).size()))

        self.feature_extractor = nn.Sequential(
            nn.Linear(conv_space, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size)
        )

    def forward(self, obs):
        # Get the convolution output
        conv_obs = self.convolution(obs).view(obs.size()[0], -1)
        # Get the random features
        features = self.feature_extractor(conv_obs)
        return features
