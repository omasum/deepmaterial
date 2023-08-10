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
from deepmaterial.utils.materialmodifier import materialmodifier_L6

@ARCH_REGISTRY.register()
class U_NETonlyHF(nn.Module):

    '''
        N_Net arch.
        input 3 decomposition of input images(3 channel)
        output 4*3 decomposition of svbrdf images(4*3 channel)
    '''
    
    def __init__(self):
        super(U_NETonlyHF, self).__init__()
        # self.unet = UNet(3,12)
        self.encoder = Encoder(10,layers=True)
        self.decodern = Decoder(3,2)
        self.decoderd = Decoder(3,3)
        self.decoderr = Decoder(3,1)
        self.decoders = Decoder(3,3)
        

    def forward(self, x):
        b, c, h, w = x.shape
        self.HighFrequency, self.dec = materialmodifier_L6.Show_subbands((x + 1.0)/2.0, Logspace=True) # [B, 8, H, W]
        self.Rchannel = self.norm(self.dec['a']) # [B, 1, H, W]
        self.Bchannel = self.norm(self.dec['b']) # [B, 1, H, W]
        self.allchannel = torch.cat([self.HighFrequency, self.Rchannel, self.Bchannel], dim=1) # [B, 10, H, W]
        self.colorcode1, self.colorcode2, self.colorcode3, self.colorcode4, self.colorcode = self.encoder(self.allchannel)
        self.normal = self.decodern(self.colorcode, self.colorcode4, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        self.diffuse = self.decoderd(self.colorcode, self.colorcode4, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        self.roughness = self.decoderr(self.colorcode, self.colorcode4, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        self.specular = self.decoders(self.colorcode, self.colorcode4, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        out = torch.cat([self.normal, self.diffuse, self.roughness, self.specular], dim=1) # [batchsize, 9, h, w]
        # out = self.unet(self.HighFrequency)
        return self.HighFrequency, out
    
    def norm(self,x):
        '''
            convert value range to [-1,1]
        '''  
        eps = 1e-20
        return 2.0*((x-torch.min(x))/(torch.max(x)-torch.min(x) + eps)) - 1.0
    

def test():
    net = U_NETonlyHF
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()