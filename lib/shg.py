import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)


class HourglassBlock(nn.Module):
    def __init__(self, channels, depth=4):
        super().__init__()
        self.depth = depth
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        for _ in range(depth):
            self.down.append(ResidualBlock(channels, channels))

        self.bottleneck = ResidualBlock(channels, channels)

        for _ in range(depth):
            self.up.append(ResidualBlock(channels, channels))

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.down[i](x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for i in reversed(range(self.depth)):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.up[i](x)
            x = x + skips[i]  # skip connection from encoder

        return x


class StackedHourglassNetwork(nn.Module):
    def __init__(self, in_channels=3, num_keypoints=68, num_stacks=2, channels=256, depth=4):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 128),
            ResidualBlock(128, channels),
        )

        self.hourglasses = nn.ModuleList([
            HourglassBlock(channels, depth=depth) for _ in range(num_stacks)
        ])

        self.out_heads = nn.ModuleList([
            nn.Conv2d(channels, num_keypoints, kernel_size=1) for _ in range(num_stacks)
        ])

        self.intermediate = nn.ModuleList()
        for _ in range(num_stacks - 1):
            self.intermediate.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Conv2d(num_keypoints, channels, kernel_size=1)
            ))

    def forward(self, x):
        x = self.preprocess(x)
        outputs = []

        for i, hg in enumerate(self.hourglasses):
            y = hg(x)
            out = self.out_heads[i](y)
            outputs.append(out)

            if i < len(self.intermediate):
                inter = self.intermediate[i]
                x = x + inter[0](y) + inter[1](out)

        return outputs  # список предсказаний с каждого hourglass
