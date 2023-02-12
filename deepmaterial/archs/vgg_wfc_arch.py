'''VGG11/13/16/19 in Pytorch.'''
import os
import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import math
from deepmaterial.utils.registry import ARCH_REGISTRY
from deepmaterial.archs.U_Net import *

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

@ARCH_REGISTRY.register()
class VGG_wfc(nn.Module):
    '''VGG network for image classification.
    for each image, output 10 numbers range [0,1]

    Args:
        vgg_name (string): Set the type of vgg network. Default: 'vgg19'.
    '''
    def __init__(self, vgg_name='VGG19'):
        super(VGG_wfc, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512*8*8, 10)
        self.unet = UNet(3,9)

    def forward(self, x):
        out = self.features(x) #[batchsize,512,hc,wc]
        out = out.view(out.size(0), -1) # [batchsize,longcharacter]
        out = self.classifier(out) # [batch_size,10]
        out = self.img_process(x,out) # image process layer, [batchsize,3,h,w]
        out = self.unet(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def fft(self,img):
        '''fourier transform of image

        Args:
            img (tensor.float[batchsize,3,h,w]): original images

        Returns:
            fshift_img(tensor.complex[batchsize,3,h,w]): images after fourier transform
        '''
       # fourier transform
        f_img = torch.fft.fft2(img) # complex64[batchsize,3,h,w]
        # shift
        fshift_img = torch.fft.fftshift(f_img)
        t_magnitude = 20*torch.log(torch.abs(fshift_img))
        # 转为numpy array
        # magnitude = t_magnitude.numpy()
        return fshift_img

    def dfft(self,m_img):
        '''dfft of passfiltered images

        Args:
            m_img (tensor.complex[batchsize,3,h,w]): frequency domain images

        Returns:
            iimg(tensor.float[batchsize,3,h,w]): images
        '''
        ishift = torch.fft.ifftshift(m_img)
        ifft = torch.fft.ifft2(ishift)
        iimg = torch.abs(ifft)
        # iimg = iimg.numpy()
        return iimg

    # filter: 10 weights of solid frequency stage
    def filter(self,m_img,weight):
        '''build filter using weight

        Args:
            m_img (tensor.float[batchsize,3,h,w]): original image, support shape for filter
            weight (tensor.float[batchsize,10]): output of classify layers

        Returns:
            tensor.float[batchsize,3,h,w]: filter
        '''
        # imgm[h,w]
        batchsize, channel, h, w = m_img.shape[0:4] #(8,3,256,256)
        # find origin
        h0,w0 = int(h/2),int(w/2) 
        # define 9 band
        bandwidth = w/2/9
        # 9 bandorigin
        bandorigin = list()
        for idx in range(1,10):
            bandorigin.append((2*idx-1)/2*bandwidth)

        # define filter
        filter = torch.zeros(size=(batchsize,h,w)) #(8,256,256)
        for i in range(0,w):
            for j in range(0,h):
                r = math.sqrt(pow(i-w0,2)+pow(j-h0,2))
                for band in bandorigin:
                    if band-bandwidth/2< r <band+bandwidth/2:
                        filter[:,i,j] = weight[:,bandorigin.index(band)]
                        break
                    filter[:,i,j] = weight[:,9]
        filter = filter.unsqueeze(1) # (8,1,256,256)
        filter = filter.expand(-1,3,-1,-1) # (8,3,256,256)
        return filter
        # filter corresponds to weights, new_magnitude corresponds to complex value

        # passfilter of images(R,G,B repectively)
    def passfilter(self,m_img,filter):
        '''passfilter for frequency domain image

        Args:
            m_img (tensor.complex[batchsize,3,h,w]): after fft image
            filter (tensor.float[batchsize,3,h,w]): filter
        Returns:
            tensor.complex[batchsize,3,h,w]: new image
        '''
        filter = filter.to('cuda')

        img = torch.mul(m_img,filter)
        # new_magnitude = 20*torch.log(torch.abs(img))
        return img
        # img corresponds to complex value

    def img_process(self,img,l_weight):
        '''filter image with wieghts learnt

        Args:
            img (tensor.float[batchsize,3,h,w]): original image 
            l_weight (tensor.float[batchsize,10])
        Return:
            image filtered(tensor.float[batchsize,3,h,w])
        '''

        new_img = torch.ones_like(img) # new batchsize image
        filter = self.filter(img,l_weight)
        f_img = self.fft(img)
        passed_img = self.passfilter(f_img,filter)
        new_img = self.dfft(passed_img)
        return new_img

def test():
    net = VGG_wfc('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())



# test()