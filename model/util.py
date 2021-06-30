import math
import os
import torch
from torch.hub import load_state_dict_from_url


def save_checkpoint(model, filename='./checkpoint.pth'):
    """Save checkpoint"""
    torch.save(model.state_dict(), filename)


def get_state_dict(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError("checkpoint file does not exist.")

    key = 'state_dict'
    checkpoint = torch.load(filename, map_location='cpu')
    if isinstance(checkpoint, dict):
        if key in checkpoint.keys():
            state_dict = checkpoint[key]
    else:
        state_dict = checkpoint
    return state_dict


def load_checkpoint(model, filename, strict=True):
    state_dict = get_state_dict(filename)
    model.load_state_dict(state_dict, strict=strict)


def adjust_conv_weights(weights, in_ch=3):
    _, ch, _, _ = weights.shape
    if(ch == in_ch):
        return weights

    wtype = weights.dtype
    weights = weights.to(torch.float32)

    if in_ch == 1 and ch == 3:
        # Sum the weights across the input channels.
        weights = weights.sum(dim=1, keepdim=True)
    elif in_ch != 3 and ch == 3:
        new_weights = torch.repeat_interleave(weights,
                                              int(math.ceil(in_ch/3)), dim=1)
        weights = new_weights[:, :ch, :, :]
        weights *= (3./in_ch)
    else:
        raise NotImplementedError("Conversion not implemented.")
    return weights.to(wtype)


def get_pretrained_weights(url, cfg, num_classes=1000, in_ch=3,
                           map_location='cpu', progress=False,
                           check_hash=False):
    state_dict = load_state_dict_from_url(url, progress=progress,
                                          map_location=map_location,
                                          check_hash=check_hash)

    conv1_name = cfg['conv1']
    classifier_name = cfg['classifier']
    model_in_ch = cfg['in_ch']
    model_num_classes = cfg['num_classes']

    if(in_ch != model_in_ch):
        try:
            state_dict[conv1_name + '.weight'] = \
                adjust_conv_weights(state_dict[conv1_name + '.weight'], in_ch)
        except NotImplementedError:
            del state_dict[conv1_name + '.weight']
            cfg['strict'] = False

    if num_classes != model_num_classes:
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        cfg['strict'] = False

    return state_dict
