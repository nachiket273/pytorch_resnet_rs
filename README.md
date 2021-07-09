# pytorch_resnet_rs
Pytorch implementation of <a href=https://arxiv.org/pdf/2103.07579.pdf>"Revisiting ResNets: Improved Training and Scaling Strategies"</a>

## Details
This repository contains pretrained weights for following models. <br>
* resnetrs50
* resnetrs101
* resnetrs152
* resnetrs200
<br>
Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs are adjusted for the implementation.<br>
<br>
Repository also contains implementation for: <br>
1) Exponential Moving Averages<br>
2) RandAugment
<br>

## Usage
### ResNetRS
1) Git clone the repoository and change to directory
```Python
git clone https://github.com/nachiket273/pytorch_resnet_rs.git
cd pytorch_resnet_rs
```

2) Import
```Python
from model import ResnetRS
```

3) List Pretrained Models
```Python
ResnetRS.list_pretrained()
```

4) Create Pretrained Model
```Python
ResnetRS.create_pretrained(model_name, in_ch=input_channels, num_classes=num_classes,
                           drop_rate=stochastic_depth_ratio)
```
5) Create Custom Model
```Python
from model.base import BasicBlock, Bottleneck
# Specify block as either BasicBlock or Bottleneck
# Specify list of number of ResBlocks as layers
# e.g layers = [3, 4, 6, 3] 
ResNetRS.create_model(block, layers, num_classes=1000, in_ch=3,
                      stem_width=64, down_kernel_size=1,
                      actn=partial(nn.ReLU, inplace=True),
                      norm_layer=nn.BatchNorm2d, seblock=True,
                      reduction_ratio=0.25, dropout_ratio=0.,
                      stochastic_depth_rate=0.0,
                      zero_init_last_bn=True)
# If you want to load custom weights
from model.util import load_checkpoint
load_checkpoint(model, filename, strict=True)
```

### Exponential Moving Averages(EMA)
1) Intialize
```Python
from model.ema import EMA
ema = EMA(model.parameters(), decay_rate=0.995, num_updates=0)
```

2) Usage in train loop
```Python
for i, (ip, tgt) in enumerate(trainloader):
    ...
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.update(model.parameters())
```

3) Usage in validation loop
```Python
for i, (ip, tgt) in enumerate(testloader):
    ...
    ema.store(model.parameters())
    ema.copy(model.parameters())
    output = model(ip)
    loss = criterion(output, tgt)
    ema.copy_back(model.parameters())
```

### RandAugment
```Python
from model.randaugment import RandAugment
raug = RandAugment(n=5, m=10)
```

## Citations

```bibtex
@misc{
    title={Revisiting ResNets: Improved Training and Scaling Strategies},
    author={Irwan Bello, William Fedus, Xianzhi Du, Ekin D. Cubuk, Aravind Srinivas, Tsung-Yi Lin, Jonathon Shlens, Barret Zoph},
    year={2021},
    url={https://arxiv.org/pdf/2103.07579.pdf}
}

@misc{
    title={RandAugment: Practical automated data augmentation with a reduced search space},
    author={Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le - Google Research, Brain Team},
    year={2019},
    url={https://arxiv.org/pdf/1909.13719v2.pdf}
}

@misc{
    title={Deep Networks with Stochastic Depth},
    author={Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Q. Weinberger},
    year={2016},
    url={https://arxiv.org/pdf/1603.09382v3.pdf}
}
```
