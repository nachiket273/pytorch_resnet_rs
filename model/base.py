""" Pytorch Resnet_RS

Building block classes for resnet.

"""
import torch
import torch.nn as nn


class StemBlock(nn.Module):
    def __init__(self, in_ch=3, stem_width=32, norm_layer=nn.BatchNorm2d,
                 actn=nn.ReLU):
        super().__init__()
        inplanes = 2 * stem_width
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(in_ch, stem_width, kernel_size=3, stride=2, padding=1,
                      bias=False),
            norm_layer(stem_width),
            actn(inplace=True),
            nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm_layer(stem_width),
            actn(inplace=True),
            nn.Conv2d(stem_width, inplanes, kernel_size=3, stride=1, padding=1,
                      bias=False)
        ])
        self.bn1 = norm_layer(inplanes)
        self.actn1 = actn(inplace=True)
        self.maxpool = nn.Sequential(*[
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1,
                      bias=False),
            norm_layer(inplanes),
            actn(inplace=True)
        ])
        self.init_weights()

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actn1(x)
        out = self.maxpool(x)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.BatchNorm2d,
                 actn=nn.ReLU, downsample=None, seblock=True,
                 reduction_ratio=0.25, stochastic_depth_ratio=0.0,
                 zero_init_last_bn=True):
        super().__init__()
        outplanes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.actn1 = actn(inplace=True)

        self.conv2 = nn.Conv2d(planes, outplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(outplanes)
        self.seblock = seblock
        if seblock:
            self.se = SEBlock(outplanes, reduction_ratio)
        self.actn2 = actn(inplace=True)
        self.down = False
        if downsample is not None:
            self.downsample = downsample
            self.down = True
        self.drop_path = DropPath(stochastic_depth_ratio)
        self.init_weights(zero_init_last_bn)

    def init_weights(self, zero_init_last_bn=True):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if zero_init_last_bn:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = self.downsample(x) if self.down else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.seblock:
            x = self.se(x)
        if self.drop_path.drop_prob > 0.0:
            x = self.drop_path(x)
        x += shortcut
        x = self.actn2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.BatchNorm2d,
                 actn=nn.ReLU, downsample=None, seblock=True,
                 reduction_ratio=0.25, stochastic_depth_ratio=0.0,
                 zero_init_last_bn=True):
        super().__init__()
        outplanes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.actn1 = actn(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.actn2 = actn(inplace=True)

        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.seblock = seblock
        if seblock:
            self.se = SEBlock(outplanes, reduction_ratio)
        self.actn3 = actn(inplace=True)
        self.down = False
        if downsample is not None:
            self.downsample = downsample
            self.down = True
        self.drop_path = DropPath(stochastic_depth_ratio)
        self.init_weights(zero_init_last_bn)

    def init_weights(self, zero_init_last_bn=True):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if zero_init_last_bn:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = self.downsample(x) if self.down else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.actn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.seblock:
            x = self.se(x)
        if self.drop_path.drop_prob:
            x = self.drop_path(x)
        x += shortcut
        x = self.actn3(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        if stride == 1:
            avgpool = nn.Identity()
        else:
            avgpool = nn.AvgPool2d(2, stride=stride, ceil_mode=True,
                                   count_include_pad=False)
        self.downsample = nn.Sequential(*[
            avgpool,
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1,
                      padding=0, bias=False),
            norm_layer(out_ch)
        ])
        self.init_weights()

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.downsample(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=0.25):
        super().__init__()
        reduced_channels = int(channels * reduction_ratio)
        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.actn = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        orig = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv1(x)
        x = self.actn(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return orig * x


class DropPath(nn.Module):
    """
    Implementation of paper https://arxiv.org/pdf/1603.09382v3.pdf
    From: https://github.com/rwightman/pytorch-image-models/blob/6d8272e92c3d5f13a9fdd91dfe1eb7fae6784589/timm/models/layers/drop.py#L160
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob is None or self.drop_prob == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],)+(1,)*(x.ndim-1)
        rand_tensor = keep_prob + torch.rand(shape, dtype=x.dtype,
                                             device=x.device)
        rand_tensor = rand_tensor.floor_()
        out = x.div(keep_prob) * rand_tensor
        return out
