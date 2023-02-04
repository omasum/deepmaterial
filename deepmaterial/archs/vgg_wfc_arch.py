'''VGG11/13/16/19 in Pytorch.'''
import os
import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import math
from deepmaterial.utils.registry import ARCH_REGISTRY

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
        self.classifier = nn.Linear(512, 10) 
        # self.filter = self.img_process()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out) # [batch_size,10]
        out = self.img_process(x,out)
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
        # 转为Tensor
        img_numpy = torch.tensor(img)
        # fourier transform
        f_img = torch.fft.fft2(img_numpy) # complex64
        # shift
        fshift_img = torch.fft.fftshift(f_img)
        t_magnitude = 20*torch.log(torch.abs(fshift_img))
        # 转为numpy array
        magnitude = t_magnitude.numpy()
        return fshift_img, magnitude

    def dfft(self,m_img):
        ishift = torch.fft.ifftshift(m_img)
        ifft = torch.fft.ifft2(ishift)
        iimg = torch.abs(ifft)
        # iimg = iimg.numpy()
        return iimg

    # filter: 10 weights of solid frequency stage
    def passfilter(self,m_img,weight):
        # imgm[h,w]
        h,w = m_img.shape[0:2] 
        # find origin
        h0,w0 = int(h/2),int(w/2) 
        # define 9 band
        bandwidth = w/2/9
        # 9 bandorigin
        bandorigin = list()
        for idx in range(1,10):
            bandorigin.append((2*idx-1)/2*bandwidth)

        # define filter
        filter = torch.zeros(size=(h,w))
        for i in range(0,w):
            for j in range(0,h):
                r = math.sqrt(pow(i-w0,2)+pow(j-h0,2))
                for band in bandorigin:
                    if band-bandwidth/2<r<band+bandwidth/2:
                        filter[i,j] = weight[bandorigin.index(band)]
                        break
                    filter[i,j] = weight[9]

        img = torch.mul(m_img,filter)
        new_magnitude = 20*torch.log(torch.abs(img))
        return filter, new_magnitude
        # filter conrrespond to weights

    def img_process(self,img,l_weight):
        '''filter image with wieghts learnt

        Args:
            img (tensor[batchsize,h,w,3]): original image 
            l_weight (tensor[batchsize,10])
        Return:
            image filtered
        '''

        batchsize = len(img)
        new_img = torch.ones_like(img) # new batchsize image
        for idx in batchsize: # each image
            image = img[idx]
            weight = l_weight[idx]
            imgr = image[:,:,2]
            imgg = image[:,:,1]
            imgb = image[:,:,0]
            fshift_imgr, magnitude_r = self.fft(imgr)
            fshift_imgg, magnitude_g = self.fft(imgg)
            fshift_imgb, magnitude_b = self.fft(imgb)
            filter, imgr_passed = self.passfilter(fshift_imgr,weight)
            _, imgg_passed = self.passfilter(fshift_imgg,weight)
            _, imgb_passed = self.passfilter(fshift_imgb,weight)
            iimgr = self.dfft(imgr_passed)
            iimgg = self.dfft(imgg_passed)
            iimgb = self.dfft(imgb_passed) #[256,256]
            new_image = torch.stack((iimgr,iimgg,iimgb),dim=2) #[256,256,3]
            new_img[idx] = new_image
        return new_img

def test():
    net = VGG_wfc('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())



# test()