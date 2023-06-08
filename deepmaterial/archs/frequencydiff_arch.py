import os
import torch
import torch.nn as nn
from deepmaterial.utils.registry import ARCH_REGISTRY
from deepmaterial.utils.materialmodifier import materialmodifier

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

@ARCH_REGISTRY.register()
class frequencydiff(nn.Module):
    '''deeplearning based band-sift to svbrdf.
    for each image, output 8*4 numbers range [0,1]

    Args:
        vgg_name (string): Set the type of vgg network. Default: 'vgg19'.
    '''
    def __init__(self, vgg_name='VGG19'):
        super(frequencydiff, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self.Classifier(512*8*8, 4*8)

    def forward(self, x):
        out = self.features(x) #[batchsize,512,hc,wc]
        out = out.view(out.size(0), -1) # [batchsize,longcharacter]
        weights = self.classifier(out) # weight: [batch_size,4*8]
        out = self.img_process(x,weights) # image process layer, [batchsize,12,h,w]
        # out = self.resize(out) # [batchsize,9,h,w] ndrs corresponding 2,3,1,3 channel

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
    
    def img_process(self, img, l_weight):
        '''filter image with wieghts learnt

        Args:
            img (tensor.float[batchsize,3,h,w]): original input of network, range(-1,1)
            l_weight (tensor.float[batchsize,4*8]): learnt weights
        Return:
            image filtered(tensor.float[batchsize,12,h,w]): sequence [n,d,r,s]
        '''

        # preprocess for image-process
        image = self.deprocess(img) # range(0,1)
        batchsize, channels, h, w = image.shape[0:4]

        # convert weight from [batchsize, 4*8] to 4 sets of weights with shape [8,batchsize,c,h,w]
        weight = torch.split(l_weight,8,dim=1) # tensors in tuple with sequence (n,d,r,s)
        n_weight = weight[0].t().unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, 1, h, w)# [8,batchsize,c,h,w]
        d_weight = weight[1].t().unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, 1, h, w)# [8,batchsize,c,h,w]
        r_weight = weight[2].t().unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, 1, h, w)# [8,batchsize,c,h,w]
        s_weight = weight[3].t().unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, 1, h, w)# [8,batchsize,c,h,w]


        maps, dec = materialmodifier.get_BS_energy(image, logspace=False)

        i = 0
        nnew_maps = {} # new maps for normal
        for key, value in maps.items():
            nnew_maps[key] = value*n_weight[i]
            i = i+1

        i = 0
        dnew_maps = {}
        for key, value in maps.items():
            dnew_maps[key] = value*d_weight[i]
            i = i+1

        i = 0
        rnew_maps = {}
        for key, value in maps.items():
            rnew_maps[key] = value*r_weight[i]
            i = i+1

        i = 0
        snew_maps = {}
        for key, value in maps.items():
            snew_maps[key] = value*s_weight[i]
            i = i+1

        normal = materialmodifier.gf_reconstruct(dec, nnew_maps, scales="BS", ind=None, logspace=False)
        diffuse = materialmodifier.gf_reconstruct(dec, dnew_maps, scales="BS", ind=None, logspace=False)
        roughness = materialmodifier.gf_reconstruct(dec, rnew_maps, scales="BS", ind=None, logspace=False)
        specular = materialmodifier.gf_reconstruct(dec, snew_maps, scales="BS", ind=None, logspace=False)

        result = torch.cat([normal, diffuse, roughness, specular], dim = 1)
        result = self.reprocess(result)

        return result  # result: filtered image[batchsize,12,h,w]


    def deprocess(self,m_img):
        '''depreocess images fed to network(-1,1) to initial images(0,1)

        Args:
            m_img (tensor.float[batchsize,3,h,w]): images fetched

        Returns:
            img (tensor.float[batchsize,3,h,w]): new image (0,1)
        '''
        # (-1~1) convert to (0~1)
        img = (m_img+1.0)/2.0

        #de-log_normalization
        img = self.de_log_normalization(img)

        return img
    
    def reprocess(self,m_img):
        '''process image filtered(0,1) to feed network(-1,1)

        Args:
            m_img (tensor.complex[batchsize,3,h,w]): images filtered
        '''
        # img = m_img / 255.0 #(0,1)
        img = 2.0 * m_img - 1.0 #(-1,1)
        return img

    def de_log_normalization(self,img):
        eps=1e-2
        image = torch.exp(img * (torch.log(1 + torch.ones((1,),device='cuda') * eps) - torch.log(torch.ones((1,),device='cuda') * eps)) + torch.log(torch.ones((1,),device='cuda') * eps))-0.01
        return image
    
    def log_normalization(self,img):
        '''log transformation of original image, dynamic range adjust, from (0,1) to (0,1)

        Args:
            img (tensor.float[batchsize,3,h,w]): original image (0,1)

        Returns:
            img (tensor.float[batchsize,3,h,w]): new image (0,1)
        '''
        eps = 1e-2
        return (torch.log(img + eps) - torch.log(torch.ones((1,),device='cuda') * eps)) / (torch.log(1 + torch.ones((1,),device='cuda') * eps) - torch.log(torch.ones((1,),device='cuda') * eps))
