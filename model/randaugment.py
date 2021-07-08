"""Pytorch RandAugment

Implementation of "RandAugment: Practical automated data augmentation with
a reduced search space" (https://arxiv.org/pdf/1909.13719v2.pdf).

"""
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import random


def contrast(img, factor):
    assert 0.1 <= factor <= 1.9, "Contrast enhancement factor should be " \
        + "in range 0.1 to 1.9"
    return ImageEnhance.Contrast(img).enhance(factor)


def brightness(img, factor):
    assert 0.1 <= factor <= 1.9, "Brightness enhancement factor should be " \
        + "in range 0.1 to 1.9"
    return ImageEnhance.Brightness(img).enhance(factor)


def color(img, factor):
    assert 0.1 <= factor <= 1.9, "Color enhancement factor should be " \
        + "in range 0.1 to 1.9"
    return ImageEnhance.Color(img).enhance(factor)


def sharpness(img, factor):
    assert 0.1 <= factor <= 1.9, "Sharpness enhancement factor should be " \
        + "in range 0.1 to 1.9"
    return ImageEnhance.Sharpness(img).enhance(factor)


def solarize(img, threshold):
    assert 0 <= threshold <= 256, "Solarize threshold should be " \
        + "within range [0, 256]"
    return ImageOps.solarize(img, threshold)


def solarizeadd(img, threshold=128, addition=0):
    assert 0 <= threshold <= 128, "SolarizeAdd threshold should be " \
        + "within range [0, 128]"
    img = np.array(img).astype(np.int64) + addition
    img = np.clip(img, 0, 255).astype(np.uint8)
    return ImageOps.solarize(img, threshold)


def autocontrast(img):
    return ImageOps.autocontrast(img)


def equalize(img):
    return ImageOps.equalize(img)


def invert(img):
    return ImageOps.invert(img)


def rotate(img, angle):
    assert 0 <= angle <= 30, "Rotation angle should be between [0, 30]"
    if random.random() > 0.5:
        angle = -angle
    return img.rotate(angle)


def posterize(img, bits):
    bits = int(bits)
    assert 4 <= bits <= 8, "The number of bits should be between [4, 8]"
    return ImageOps.posterize(img, bits)


def shearx(img, level):
    assert 0 <= level <= 0.3, "Shear level should be in range [0, 0.3]"
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


def sheary(img, level):
    assert 0 <= level <= 0.3, "Shear level should be in range [0, 0.3]"
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


def translatex(img, percentage):
    assert 0.0 <= percentage <= 0.45, "translate percentage should be " \
          + "between [0.0, 0.45]"
    if random.random() > percentage:
        percentage = -percentage
    level = img.size[0] * percentage
    return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


def translatey(img, percentage):
    assert 0.0 <= percentage <= 0.45, "translate percentage should be " \
          + "between [0.0, 0.45]"
    if random.random() > percentage:
        percentage = -percentage
    level = img.size[1] * percentage
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


def cutout(img, percentage):
    assert 0.0 <= percentage <= 0.2, "cutout percentage should be " \
        + "within range [0.0, 0.2]"
    if percentage < 0:
        return img
    level = img.size[0] * percentage
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - level / 2.))
    y0 = int(max(0, y0 - level / 2.))
    x1 = min(w, x0 + level)
    y1 = min(h, y0 + level)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


NAME_TO_FUNC = {
    "AutoContrast": autocontrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "Solarize": solarize,
    "SolarizeAdd": solarizeadd,
    "Color": color,
    "Contrast": contrast,
    "Brightness": brightness,
    "Sharpness": sharpness,
    "ShearX": shearx,
    "ShearY": sheary,
    "TranslateX": translatex,
    "TranslateY": translatey,
    "CutOut": cutout
}


# Augmentation list from
# https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
AVAILABLE_AUGS = [
    ["AutoContrast", -1, -1],
    ["Equalize", -1, -1],
    ["Invert", -1, -1],
    ["Rotate", 0, 30],
    ["Posterize", 4, 8],
    ["Solarize", 0, 256],
    ["Color", 0.1, 1.9],
    ["Contrast", 0.1, 1.9],
    ["Brightness", 0.1, 1.9],
    ["Sharpness", 0.1, 1.9],
    ["ShearX", 0.0, 0.3],
    ["ShearY", 0.0, 0.3],
    ["TranslateX", 0.0, 0.45],
    ["TranslateY", 0.0, 0.45],
    ["CutOut", 0.0, 0.2],
    ["SolarizeAdd", 0, 128]
]

MIN_LEVEL = 0
MAX_LEVEL = 30


class RandAugment:
    def __init__(self, n: int, m: int) -> None:
        self.n = n
        self.m = m

    def __call__(self, img) -> Image:
        ops = np.random.choice(range(len(AVAILABLE_AUGS)), self.n)
        for i in ops:
            name, min_lvl, max_lvl = AVAILABLE_AUGS[i]
            func = NAME_TO_FUNC[name]
            if name in ["AutoContrast", "Equalize", "Invert"]:
                img = func(img)
            else:
                val = (float(self.m) / (MAX_LEVEL - MIN_LEVEL))
                val = val * float(max_lvl - min_lvl) + min_lvl
                if name == "SolarizeAdd":
                    addition = 7
                    img = func(img, val, addition)
                else:
                    img = func(img, val)

        return img
