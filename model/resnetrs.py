"""Pytorch Resnet_RS

This file contains pytorch implementation of Resnet_RS architecture from paper
"Revisiting ResNets: Improved Training and Scaling Strategies"
(https://arxiv.org/pdf/2103.07579.pdf)

"""
import torch.nn as nn

from base import StemBlock, BasicBlock, Bottleneck, Downsample


class ResnetRS(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_ch=3, stem_width=64,
                 down_kernel_size=1, actn=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 zero_init_last_bn=True):
        super().__init__()
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.actn = actn
        self.zero_init_last_bn = zero_init_last_bn
        self.conv1 = StemBlock(in_ch, stem_width, norm_layer, actn)
        channels = [64, 128, 256, 512]
        self.make_layers(block, layers, channels, stem_width*2,
                         down_kernel_size)
        self.avg_pool = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        ])
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=True)

    def make_layers(self, block, nlayers, channels, inplanes, kernel_size=1):
        for idx, (nlayer, channel) in enumerate(zip(nlayers, channels)):
            name = "layer" + str(idx+1)
            stride = 1 if idx == 0 else 2
            downsample = None
            if stride != 1 or inplanes != channel * block.expansion:
                downsample = Downsample(inplanes, channel * block.expansion,
                                        kernel_size=kernel_size, stride=stride,
                                        norm_layer=self.norm_layer)

            blocks = []
            for layer_idx in range(nlayer):
                downsample = downsample if layer_idx == 0 else None
                stride = stride if layer_idx == 0 else 1
                blocks.append(block(inplanes, channel, stride, self.norm_layer,
                                    self.actn, downsample,
                                    self.zero_init_last_bn))

                inplanes = channel * block.expansion

            self.add_module(*(name, nn.Sequential(*blocks)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x
