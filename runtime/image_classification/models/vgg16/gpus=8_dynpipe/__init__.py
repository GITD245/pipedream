# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .vgg16 import VGG16Split
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2

def arch():
    return "vgg16"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0"]),
        (Stage1(), ["out0"], ["out1"]),
        (Stage2(), ["out1"], ["out2"]),
        (criterion, ["out2"], ["loss"])
    ]

def full_model():
    return VGG16Split()
