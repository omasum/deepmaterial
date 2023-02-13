import os
import torch


weight = torch.tensor([[2,1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1,5]])
image = torch.randn((2,3,256,256))


def filter(m_img,weight):
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
            m = torch.nn.ConstantPad2d(bandwidth,weight[i,j].item) 
            subfilter = m(subfilter) # enlarge subfilter and fill weight[i,j]
        m_10 = torch.nn.ConstantPad2d(bandwidth_10,weight[i,9].item)
        subfilter = m_10(subfilter) # [256,256]
        subfilter = subfilter.unsqueeze(0) #[1,256,256]
        subfilter = subfilter.expand(channel,-1,-1) #[3,256,256]
        subfilter = subfilter.unsqueeze(0) #[1,3,256,256]
        unfilter.append(subfilter)
    filter = torch.cat(unfilter,dim=0) #[8,3,256,256]

    return filter
print(image)
print(filter(m_img=image,weight=weight))
print('success')