
import os
import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import math
from deepmaterial.utils.registry import ARCH_REGISTRY
from deepmaterial.archs.U_Net import *

@ARCH_REGISTRY.register()
class Refine_specular(nn.Module):

    '''
        U_Net arch.
        input pred svbrdf map(3 channel) and chosen subband of the input image
        output refined svbrdf map(3 channel)
    '''
    
    def __init__(self):
        super(Refine_specular, self).__init__()
        # self.unet = UNet(3,12)
        self.unet = UNet(4, 3)
        

    def forward(self, x):
        out = self.unet(x)
        return out