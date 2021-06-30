"""Pytorch Resnet_RS

This file contains pytorch implementation of Resnet_RS architecture from paper
"Revisiting ResNets: Improved Training and Scaling Strategies"
(https://arxiv.org/pdf/2103.07579.pdf)

"""
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

from base import StemBlock, BasicBlock, Bottleneck, Downsample
from util import get_pretrained_weights


PRETRAINED_MODELS = [
    'resnetrs50',
    'resnetrs101',
    'resnetrs152',
    'resnetrs200'
]

PRETRAINED_URLS = {
    'resnetrs50': '',
    'resnetrs101': '',
    'resnetrs152': '',
    'resnetrs200': '',
}

DEFAULT_CFG = {
    'in_ch': 3,
    'num_classes': 1000,
    'stem_width': 64,
    'down_kernel_size': 1,
    'actn': partial(nn.ReLU, inplace=True),
    'norm_layer': nn.BatchNorm2d,
    'zero_init_last_bn': True,
    'seblock': True,
    'reduction_ratio': 0.25,
    'dropout_ratio': 0.,
    'conv1': 'conv1',
    'classifier': 'fc'
}


class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_ch=3, stem_width=64,
                 down_kernel_size=1, actn=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 seblock=True, reduction_ratio=0.25, dropout_ratio=0.,
                 zero_init_last_bn=True):
        super().__init__()
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.actn = actn
        self.dropout_ratio = float(dropout_ratio)
        self.zero_init_last_bn = zero_init_last_bn
        self.conv1 = StemBlock(in_ch, stem_width, norm_layer, actn)
        channels = [64, 128, 256, 512]
        self.make_layers(block, layers, channels, stem_width*2,
                         down_kernel_size, seblock, reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=True)

    def make_layers(self, block, nlayers, channels, inplanes, kernel_size=1,
                    seblock=True, reduction_ratio=0.25):
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
                                    self.actn, downsample, seblock,
                                    reduction_ratio, self.zero_init_last_bn))

                inplanes = channel * block.expansion

            self.add_module(*(name, nn.Sequential(*blocks)))

    def init_weights(self, zero_init_last_bn=True):
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
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(1, -1)
        if self.dropout_ratio > 0.:
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.fc(x)
        return x


class ResnetRS():
    def __init__(self):
        super().__init__()

    @classmethod
    def create_model(cls, block, layers, num_classes=1000, in_ch=3,
                     stem_width=64, down_kernel_size=1,
                     actn=partial(nn.ReLU, inplace=True),
                     norm_layer=nn.BatchNorm2d, seblock=True,
                     reduction_ratio=0.25, dropout_ratio=0.,
                     zero_init_last_bn=True):

        return Resnet(block, layers, num_classes=num_classes, in_ch=in_ch,
                      stem_width=stem_width, down_kernel_size=down_kernel_size,
                      actn=actn, norm_layer=norm_layer, seblock=seblock,
                      reduction_ratio=reduction_ratio,
                      dropout_ratio=dropout_ratio,
                      zero_init_last_bn=zero_init_last_bn)

    @classmethod
    def list_pretrained(cls):
        return PRETRAINED_MODELS

    @classmethod
    def _is_valid_model_name(cls, name):
        name = name.strip()
        name = name.lower()
        return name in PRETRAINED_MODELS

    @classmethod
    def _get_url(cls, name):
        return PRETRAINED_URLS[name]

    @classmethod
    def _get_default_cfg(cls):
        return DEFAULT_CFG

    @classmethod
    def _get_cfg(cls, name):
        cfg = ResnetRS._get_default_cfg()
        cfg['block'] = Bottleneck
        if name == 'resnetrs50':
            cfg['layers'] = [3, 4, 6, 3]
        elif name == 'resnetrs101':
            cfg['layers'] = [3, 4, 23, 3]
        elif name == 'resnetrs152':
            cfg['layers'] = [3, 8, 36, 3]
        elif name == 'resnetrs200':
            cfg['layers'] = [3, 24, 36, 3]
        return cfg

    @classmethod
    def create_pretrained(cls, name, in_ch=0, num_classes=0):
        if not ResnetRS._is_valid_model_name(name):
            raise ValueError('Available pretrained models: ' +
                             ', '.join(PRETRAINED_MODELS))

        cfg = ResnetRS._get_cfg(name)
        in_ch = cfg['in_ch'] if in_ch == 0 else in_ch
        num_classes = cfg['num_classes'] if num_classes == 0 else num_classes

        url = ResnetRS._get_url(name)
        model = Resnet(cfg['block'], cfg['layers'], num_classes=num_classes,
                       in_ch=in_ch, stem_width=cfg['stem_width'],
                       down_kernel_size=cfg['down_kernel_size'],
                       actn=cfg['actn'], norm_layer=cfg['norm_layer'],
                       seblock=cfg['seblock'],
                       reduction_ratio=cfg['reduction_ratio'],
                       zero_init_last_bn=cfg['zero_init_last_bn'])

        state_dict = get_pretrained_weights(url, cfg, num_classes, in_ch,
                                            check_hash=True)

        model.load_state_dict(state_dict, strict=cfg['strict'])
        return model
