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
class VGG_wfc_sf(nn.Module):
    '''VGG network for image classification.
    for each image, output 10 numbers range [0,1]

    Args:
        vgg_name (string): Set the type of vgg network. Default: 'vgg19'.
    '''
    def __init__(self, vgg_name='VGG19'):
        super(VGG_wfc_sf, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self.Classifier(512*8*8, 40)
        self.unetn = UNet(3,2)
        self.unetd = UNet(3,3)
        self.unetr = UNet(3,1)
        self.unets = UNet(3,3)

    def forward(self, x):
        out = self.features(x) #[batchsize,512,hc,wc]
        out = out.view(out.size(0), -1) # [batchsize,longcharacter]
        out = self.classifier(out) # weight: [batch_size,40]
        out = self.img_process(x,out) # image process layer, [batchsize,12,h,w]
        outn = self.unetn(out[:,0:3]) # [batchsize,2,h,w]
        outd = self.unetd(out[:,3:6]) # [batchsize,3,h,w]
        outr = self.unetr(out[:,6:9]) # [batchsize,1,h,w]
        outs = self.unets(out[:,9:12]) # [batchsize,3,h,w]
        out = torch.cat([outn,outd,outr,outs],dim=1) # [batchsize,9,h,w]

        return out
    
    def Classifier(self,in_chnnel,out_chnnel):
        layers = []
        layers += [nn.Linear(in_chnnel,out_chnnel)]
        layers += [nn.Sigmoid()] # output weight must in (0,1)
        return nn.Sequential(*layers)

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
            weight (tensor.float[batchsize,10]): output of classify layers, 3 channels of an image share the same weights however each image has different weights

        Returns:
            tensor.float[batchsize,3,h,w]: filter
        '''
        # imgm[h,w]
        batchsize, channel, h, w = m_img.shape[0:4] #(8,3,256,256)
        # find origin
        h0,w0 = int(h/2),int(w/2) # 128,128
        # define 1-9 bandwidth
        bandwidth = int(w/2//9) # 14*9+2=256

        # define 10 bandwidth
        bandwidth_10 = int(w/2%9) # 14*9+2=256

        # define filter
        unfilter = []
        for i in range(0,batchsize): # 0-7
            subfilter = torch.zeros(bandwidth*2,bandwidth*2) #(28,28)
            subfilter[:] = weight[i,0]
            for j in range(1,9): # 1-9
                m = torch.nn.ConstantPad2d(bandwidth,weight[i,j].item()) 
                subfilter = m(subfilter) # enlarge subfilter and fill weight[i,j]
            m_10 = torch.nn.ConstantPad2d(bandwidth_10,weight[i,9].item())
            subfilter = m_10(subfilter) # [256,256]
            subfilter = subfilter.unsqueeze(0) #[1,256,256]
            subfilter = subfilter.expand(channel,-1,-1) #[3,256,256]
            subfilter = subfilter.unsqueeze(0) #[1,3,256,256]
            unfilter.append(subfilter)
        filter = torch.cat(unfilter,dim=0) #[8,3,256,256]

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

    def deprocess(self,m_img):
        '''depreocess images fed to network(-1,1) to initial images(0,255)

        Args:
            m_img (tensor.complex[batchsize,3,h,w]): images fetched
        '''
        # (-1~1) convert to (0~1)
        img = (m_img+1.0)/2.0

        #de-log_normalization
        img = self.de_log_normalization(img)

        #(0,1) convert to (0,255)
        img = img * 255.0
        return img
    
    def reprocess(self,m_img):
        '''process image filtered(0,255) to feed network(-1,1)

        Args:
            m_img (tensor.complex[batchsize,3,h,w]): images filtered
        '''
        img = m_img / 255.0 #(0,1)
        img = 2.0 * img - 1.0 #(-1,1)
        return img

    def de_log_normalization(self,img):
        eps=1e-2
        image = torch.exp(img * (torch.log(1 + torch.ones((1,),device='cuda') * eps) - torch.log(torch.ones((1,),device='cuda') * eps)) + torch.log(torch.ones((1,),device='cuda') * eps))-0.01
        return image
    
    def log_normalization(self,img):
        eps = 1e-2
        return (torch.log(img + eps) - torch.log(torch.ones((1,),device='cuda') * eps)) / (torch.log(1 + torch.ones((1,),device='cuda') * eps) - torch.log(torch.ones((1,),device='cuda') * eps))

    def img_process(self,img,l_weight):
        '''filter image with wieghts learnt

        Args:
            img (tensor.float[batchsize,3,h,w]): original input 
            l_weight (tensor.float[batchsize,40])
        Return:
            image filtered(tensor.float[batchsize,12,h,w]): sequence [n,d,r,s]
        '''
        image = self.deprocess(img)
        weight = torch.split(l_weight,10,dim=1) # tensors in tuple with sequence (n,d,r,s)

        consolidate = []
        new_img = torch.ones_like(image) # new batchsize image
        
        for w in weight:
            filter = self.filter(image,w)
            f_img = self.fft(image)
            passed_img = self.passfilter(f_img,filter)
            new_img = self.dfft(passed_img)

            new_img = self.reprocess(new_img)
            consolidate.append(new_img) # filtered image[n,d,r,s],[4tenror(batchsize,3,h,w)]
        
        result = torch.cat(consolidate,dim=1)
        return result # filtered image[batchsize,12,h,w]
    
    def visualization(self,filter):
        '''save filter

        Args:
            filter (tensor.float[batchsize,3,h,w]): learned filter
        '''
        # can use tensor2img in img_util.py
        pass

def test():
    net = VGG_wfc_sf('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())



# test()