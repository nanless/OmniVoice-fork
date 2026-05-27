# SimAM ResNet100 + ASP pooling (voxblink2_samresnet100 architecture).
# Minimal vendored subset for eval_sim; no external wespeaker dependency.

from __future__ import annotations

import torch
import torch.nn as nn


class ASP(nn.Module):
    def __init__(self, in_planes: int, acoustic_dim: int):
        super().__init__()
        outmap_size = int(acoustic_dim / 8)
        self.out_dim = in_planes * 8 * outmap_size * 2
        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1, x.size(-1))
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        return torch.cat((mu, sg), 1).view(x.size(0), -1)


class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self._simam(out) + self.downsample(x)
        return self.relu(out)

    def _simam(self, x, lambda_p=1e-4):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + lambda_p)) + 0.5
        return x * self.sigmoid(e_inv)


class SimAMResNet100(nn.Module):
    def __init__(self, in_planes=64, in_ch=1):
        super().__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(in_ch, in_planes, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, [6, 1])
        self.layer2 = self._make_layer(128, [16, 2])
        self.layer3 = self._make_layer(256, [24, 2])
        self.layer4 = self._make_layer(512, [3, 2])

    def _make_layer(self, planes, num_blocks_stride):
        num_blocks, stride = num_blocks_stride
        layers = [SimAMBasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(SimAMBasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class SimAMResNet100ASP(nn.Module):
    """Speaker embedding model: SimAM_ResNet100 + ASP."""

    def __init__(self, embed_dim=256, in_planes=64, acoustic_dim=80, dropout=0):
        super().__init__()
        self.front = SimAMResNet100(in_planes)
        self.pooling = ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.front(x)
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        return self.bottleneck(x)
