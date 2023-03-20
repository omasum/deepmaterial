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

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

@ARCH_REGISTRY.register()
class VGG_dwt_sf(nn.Module):
    '''VGG network for image classification.
    for each image, output 10 numbers range [0,1]

    Args:
        vgg_name (string): Set the type of vgg network. Default: 'vgg19'.
    '''
    def __init__(self, vgg_name='VGG19'):
        super(VGG_dwt_sf, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self.Classifier(512*8*8, 16)
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

    def dwt(self,img):
        '''Haar DWT of image

        Args:
            img (tensor.float[batchsize,3,h,w]): original images

        Returns:
            dwt_img(tensor.float[batchsize,c*4,h/2,w/2]):return cat of 4 feature maps
        '''
     
        x01 = img[:, :, 0::2, :] / 2 # odd row
        x02 = img[:, :, 1::2, :] / 2 # even row
        x1 = x01[:, :, :, 0::2] # odd row, odd column
        x2 = x02[:, :, :, 0::2] # even row, odd column
        x3 = x01[:, :, :, 1::2] # odd row, even column
        x4 = x02[:, :, :, 1::2] # even row, even column
        x_LL = x1 + x2 + x3 + x4 # height/2, width/2
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

    def idwt(self,m_img):
        ''' Haar iDWT of features

        Args:
            m_img (tensor.float[batchsize,c*4,h/2,w/2]): wavelet features

        Returns:
            iimg(tensor.float[batchsize,3,h,w]): images
        '''
        r = 2
        in_batch, in_channel, in_height, in_width = m_img.size()
        #print([in_batch, in_channel, in_height, in_width])
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = m_img[:, 0:out_channel, :, :] / 2
        x2 = m_img[:, out_channel:out_channel * 2, :, :] / 2
        x3 = m_img[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = m_img[:, out_channel * 3:out_channel * 4, :, :] / 2
        

        # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

    def Filter(self,m_img,weight):
        '''build filter using weight

        Args:
            m_img (tensor.float[batchsize,c*4,h/2,w/2]): image after dwt, support shape for filter
            weight (tensor.float[batchsize,4]): output of classify layers, 3 channels of an image share the same weights while each image has different weights

        Returns:
            tensor.float[batchsize,c*4,h/2,w/2]: filter
        '''
        batchsize, channels, h, w = m_img.shape[0:4] #(8,3*4,256/2,256/2)
        channel = int(channels/4)
        weight = weight.unsqueeze(-1)
        weight = weight.unsqueeze(-1) # [batchsize,4,1,1]
        llw, hlw, lhw, hhw = weight.split(1, dim=1) # tuple of [batchsize,1,1,1]
        ll_weight = llw.expand(-1, channel, h, w)
        hl_weight = hlw.expand(-1, channel, h, w)
        lh_weight = lhw.expand(-1, channel, h, w)
        hh_weight = hhw.expand(-1, channel, h, w)
        filter = torch.cat([ll_weight, hl_weight, lh_weight, hh_weight], dim = 1)
        return filter

        # passfilter of images(R,G,B repectively)
    def passfilter(self,m_img,filter):
        '''passfilter for frequency domain image

        Args:
            m_img (tensor.float[batchsize, 4*3, h/2, w/2]): after dwt image
            filter (tensor.float[batchsize, 4*3, h/2, w/2]): filter for LL, HL, LH, HH features
        Returns:
            tensor.float[batchsize,4*3,h/2,w/2]: new wavelet features of each svbrdf maps
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
            l_weight (tensor.float[batchsize,4*4])
        Return:
            image filtered(tensor.float[batchsize,4*4*3,h,w]): sequence [n,d,r,s]
        '''
        image = self.deprocess(img)
        weight = torch.split(l_weight,4,dim=1) # tensors in tuple with sequence (n,d,r,s)

        consolidate = []
        new_img = torch.ones_like(image) # new batchsize image
        
        for w in weight:
            f_img = self.dwt(image)
            filter = self.Filter(f_img,w)
            passed_img = self.passfilter(f_img,filter)
            new_img = self.idwt(passed_img)

            new_img = self.reprocess(new_img)
            consolidate.append(new_img) # filtered image[n,d,r,s],[4tenror(batchsize,3,h,w)]
        
        result = torch.cat(consolidate,dim=1)
        result = result.to('cuda')
        return result # filtered image[batchsize,12,h,w]
    
    def visualization(self,filter):
        '''save filter

        Args:
            filter (tensor.float[batchsize,3,h,w]): learned filter
        '''
        # can use tensor2img in img_util.py
        pass

def test():
    net = VGG_dwt_sf('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())



# test()