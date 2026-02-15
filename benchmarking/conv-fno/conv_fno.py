"""
###############################################################################
# Benchmarking Python Module (2026)
#
# Author: Vittoria De Pellegrini
# Affiliation: PhD Student, King Abdullah University of Science and Technology
#
# Description: conv-fno/conv_fno.py
#              Baseline benchmarking source file for training, modeling,
#              evaluation, or utilities in the benchmarking workflow.
###############################################################################
"""

import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        """3D Fourier layer: FFT -> linear transform -> iFFT."""
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class ConvFNOBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width

        self.fc0 = nn.Linear(12, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)

        self.c3 = nn.Conv3d(self.width, self.width, kernel_size=3, padding=1)
        self.c4 = nn.Conv3d(self.width, self.width, kernel_size=3, padding=1)
        self.c5 = nn.Conv3d(self.width, self.width, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def _fourier_layer(self, x, spectral, bias):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[2], x.shape[3], x.shape[4]

        x1 = spectral(x)
        x2 = bias(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.relu(x)
        return x

    def _conv_fourier_layer(self, x, spectral, bias, conv):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[2], x.shape[3], x.shape[4]

        x1 = spectral(x)
        x2 = bias(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = conv(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        return x

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = self._fourier_layer(x, self.conv0, self.w0)
        x = self._fourier_layer(x, self.conv1, self.w1)
        x = self._fourier_layer(x, self.conv2, self.w2)
        x = self._conv_fourier_layer(x, self.conv3, self.w3, self.c3)
        x = self._conv_fourier_layer(x, self.conv4, self.w4, self.c4)
        x = self._conv_fourier_layer(x, self.conv5, self.w5, self.c5)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class ConvFNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super().__init__()
        self.block = ConvFNOBlock3d(modes1, modes2, modes3, width)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        x = F.pad(F.pad(x, (0, 0, 0, 8, 0, 8), "replicate"), (0, 0, 0, 0, 0, 0, 0, 8), "constant", 0)
        x = self.block(x)
        x = x.view(batchsize, size_x + 8, size_y + 8, size_z + 8, 1)[..., :-8, :-8, :-8, :]
        return x.squeeze()

    def count_params(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())
