from typing import Optional, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter






class Net(nn.Module):
    def __init__(self,fc_dim,nb_classes):
        super(Net, self).__init__()


        # Define the network components
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True)
        )


        # self.fc1 = nn.Linear(fc_dim,fc_dim//100)
        # self.fc2 = nn.Linear(fc_dim//100,1)

        self.fc = nn.Linear(fc_dim,nb_classes)

    def forward(self, x):

        # # shift and scale input to mean=0 std=1 (across all bins)
        # x = x + self.input_mean
        # x = x * self.input_scale

        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)


        x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = self.fc2(x)

        output = self.fc(x).squeeze()
        # output = F.log_softmax(x, dim=1)

        # output = torch.sigmoid(x).squeeze()

        return output        