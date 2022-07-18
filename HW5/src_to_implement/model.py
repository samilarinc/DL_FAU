import torch.nn as nn
# from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.LazyConv2d(out_channels, 1, stride=stride)
        self.NN = nn.Sequential(
            nn.LazyConv2d(out_channels, 3, stride, 1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, 3, padding=1),
            nn.LazyBatchNorm2d()
        )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        Y = self.NN(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return self.ReLU(x + Y)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.NN = nn.Sequential(nn.Conv2d(3, 64, 7, 2),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(),
                                nn.MaxPool2d(3, 2),
                                ResBlock(64, 64, 1),
                                ResBlock(64, 128, 2),
                                ResBlock(128, 256, 2),
                                ResBlock(256, 512, 2),
                                nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten(),
                                nn.LazyLinear(2),
                                nn.Sigmoid()
        )

    def forward(self, x):
        return self.NN(x)
        
# summary(ResNet().cuda(), (3, 300, 300))