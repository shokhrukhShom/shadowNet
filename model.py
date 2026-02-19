"""
ShadowNet – U-Net model for vehicle shadow generation.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """4-level U-Net: 1-ch grayscale mask → 1-ch shadow map."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.d1 = DoubleConv(1, 32)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(64, 128)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(128, 256)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = DoubleConv(256, 512)

        # Decoder
        self.u1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c1 = DoubleConv(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = DoubleConv(256, 128)
        self.u3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c3 = DoubleConv(128, 64)
        self.u4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.c4 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))

        mid = self.mid(self.p4(d4))

        u1 = self.c1(torch.cat([self.u1(mid), d4], 1))
        u2 = self.c2(torch.cat([self.u2(u1), d3], 1))
        u3 = self.c3(torch.cat([self.u3(u2), d2], 1))
        u4 = self.c4(torch.cat([self.u4(u3), d1], 1))

        return torch.sigmoid(self.out(u4))
