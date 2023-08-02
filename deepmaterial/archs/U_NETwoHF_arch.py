'''VGG11/13/16/19 in Pytorch.'''
# add learned fourier filter to get 4 maps splitly
import os
import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import math
from deepmaterial.utils.registry import ARCH_REGISTRY
from deepmaterial.archs.U_Net import *
from pytorch_wavelets import DWTForward, DWTInverse

@ARCH_REGISTRY.register()
class U_NETwoHF(nn.Module):

    '''
        N_Net arch.
        input 3 decomposition of input images(3 channel)
        output 4*3 decomposition of svbrdf images(4*3 channel)
    '''
    
    def __init__(self):
        super(U_NETwoHF, self).__init__()
        # self.unet = UNet(3,12)
        self.encoder = Encoder(3,layers=True)
        self.decodern = Decoder(3,2)
        self.decoderd = Decoder(3,3)
        self.decoderr = Decoder(3,1)
        self.decoders = Decoder(3,3)
        

    def forward(self, x):
        b, c, h, w = x.shape
        self.HighFrequency = torch.ones(b, c, int(h/2), int(w/2))
        self.colorcode1, self.colorcode2, self.colorcode3, self.colorcode4, self.colorcode = self.encoder(x)
        self.normal = self.decodern(self.colorcode, self.colorcode4, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        self.diffuse = self.decoderd(self.colorcode, self.colorcode4, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        self.roughness = self.decoderr(self.colorcode, self.colorcode4, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        self.specular = self.decoders(self.colorcode, self.colorcode4, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        out = torch.cat([self.normal, self.diffuse, self.roughness, self.specular], dim=1) # [batchsize, 9, h, w]
        # out = self.unet(self.HighFrequency)
        return self.HighFrequency, out
    

def test():
    net = U_NETwoHF
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()