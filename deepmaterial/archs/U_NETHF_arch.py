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
# from pytorch_wavelets import DWTForward, DWTInverse
from deepmaterial.utils.materialmodifier import materialmodifier_L6

@ARCH_REGISTRY.register()
class U_NETHF(nn.Module):

    '''
        N_Net arch.
        input 3 decomposition of input images(3 channel)
        output 4*3 decomposition of svbrdf images(4*3 channel)
    '''
    
    def __init__(self):
        super(U_NETHF, self).__init__()
        # self.unet = UNet(3,12)
        self.encodern = subEncoder(8, layers = True, bilinear=True)
        self.encoderd = subEncoder(8, layers = True, bilinear=True)
        self.encoderr = subEncoder(8, layers = True, bilinear=True)
        self.encoders = subEncoder(8, layers = True, bilinear=True)
        self.encoder = subEncoder(3, layers=True, bilinear=True)
        self.decodern = subDecoder(3, 2)
        self.decoderd = subDecoder(3, 3)
        self.decoderr = subDecoder(3, 1)
        self.decoders = subDecoder(3, 3)
        self.fusionn = SELayer(512)
        self.fusiond = SELayer(512)
        self.fusionr = SELayer(512)
        self.fusions = SELayer(512)
        

    def forward(self, x):
        self.HighFrequency, self.dec = materialmodifier_L6.Show_subbands((x + 1.0)/2.0, Logspace=True) # [batchsize, 8, H, W]
        _, _, _, self.ncode = self.encodern(self.HighFrequency) # [batchsize, 256, 16, 16]
        _, _, _, self.dcode = self.encoderd(self.HighFrequency) # [batchsize, 256, 16, 16]
        _, _, _, self.rcode = self.encoderr(self.HighFrequency) # [batchsize, 256, 16, 16]
        _, _, _, self.scode = self.encoders(self.HighFrequency) # [batchsize, 256, 16, 16]
        self.colorcode1, self.colorcode2, self.colorcode3, self.colorcode = self.encoder(x)
        self.final_ncode = self.fusionn(torch.cat([self.ncode, self.colorcode], dim=1)) #[B, 512, 16, 16]
        self.final_dcode = self.fusiond(torch.cat([self.dcode, self.colorcode], dim=1)) #[B, 512, 16, 16]
        self.final_rcode = self.fusionr(torch.cat([self.rcode, self.colorcode], dim=1)) #[B, 512, 16, 16]
        self.final_scode = self.fusions(torch.cat([self.scode, self.colorcode], dim=1)) #[B, 512, 16, 16]
        self.normal = self.decodern(self.final_ncode, self.colorcode3, self.colorcode2, self.colorcode1)
        self.diffuse = self.decoderd(self.final_dcode, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        self.roughness = self.decoderr(self.final_rcode, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        self.specular = self.decoders(self.final_scode, self.colorcode3, self.colorcode2, self.colorcode1) # [batchsize, 3, h, w]
        out = torch.cat([self.normal, self.diffuse, self.roughness, self.specular], dim=1) # [batchsize, 9, h, w]
        # out = self.unet(self.HighFrequency)
        return self.HighFrequency, out
    

    # def Get_HighFrequency(self, x):
    #     '''
    #         get high frequency decompositions from processed input[B, 3, H, W] ranges(-1,1) to high coefficient
    #     '''
    #     img = self.deprocess(x) # [B, 3, H, W]
    #     img = self.grayImg(img) # [B, 1, H, W]
    #     self.dwt = DWTForward(J=1, wave='haar', mode='zero')
    #     self.idwt = DWTInverse(wave='haar', mode='zero')
    #     f_inputl, f_inputh = self.dwt(img.cpu()) # f_inputh is a list with level elements

    #     HighFrequency = f_inputh[0].cuda() # [B, 1, 3, H/2, W/2]
    #     HighFrequency = torch.cat([HighFrequency[:,:,0,:,:],HighFrequency[:,:,1,:,:],HighFrequency[:,:,2,:,:]], dim=1) # [B, 3, H/2, W/2]
    #     HighFrequency = self.norm(HighFrequency)
    #     return HighFrequency

    def de_gamma(self,img):
        
        image = img**2.2
        return image

    def norm(self,x):
        '''
            convert value range to [-1,1]
        '''  
        return 2.0*((x-torch.min(x))/(torch.max(x)-torch.min(x))) - 1.0
    
    def deprocess(self,m_img):
        '''depreocess images fed to network(-1,1) to initial images(0,255)

        Args:
            m_img (tensor.complex[batchsize,3,h,w]): images fetched
        '''
        # (-1~1) convert to (0~1)
        img = (m_img+1.0)/2.0

        #de-log_normalization
        img = self.de_gamma(img)

        #(0,1) convert to (0,255)
        img = img * 255.0
        return img
    
    def grayImg(self, img):
        '''
            turn RGB images [B, 3, H, W] to Gray images, range(0,255), size[B, 1, H, W]
        '''
        image = torch.unsqueeze(1/3*img[:,0,:,:] + 1/3*img[:,1,:,:] + 1/3*img[:,2,:,:], dim=1)
        return image


def test():
    net = U_NETHF
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()