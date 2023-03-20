import math
import os



import cv2
from matplotlib import pyplot as plt 

import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def dwt_init(x):
    '''
        Haar DWT of x[batchsize,c,h,w], return cat of 4 feature maps[batchsize,c*4,h/2,w/2]
    '''

    x01 = x[:, :, 0::2, :] / 2 # odd row
    x02 = x[:, :, 1::2, :] / 2 # even row
    x1 = x01[:, :, :, 0::2] # odd row, odd column
    x2 = x02[:, :, :, 0::2] # even row, odd column
    x3 = x01[:, :, :, 1::2] # odd row, even column
    x4 = x02[:, :, :, 1::2] # even row, even column
    x_LL = x1 + x2 + x3 + x4 # height/2, width/2
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):

    '''
        Haar iDWT of features[batchsize,c*4,h/2,w/2], return h[batchsize,c,h,w]
    
    '''
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
    
def show(subbands):

    '''
        subbands[batchsize,c*4,h',w']
        show LL, LH, HL, HH features in 4 plots
    '''

    title = ['LL','HL','LH','HH']

    plt.subplot(2,2,1), plt.imshow(subbands[0]), plt.title(title[0]), plt.axis('off')
    plt.subplot(2,2,2), plt.imshow(subbands[1]), plt.title(title[1]), plt.axis('off')
    plt.subplot(2,2,3), plt.imshow(subbands[2]), plt.title(title[2]), plt.axis('off')
    plt.subplot(2,2,4), plt.imshow(subbands[3]), plt.title(title[3]), plt.axis('off')
    plt.show()
    
def test(dwt, idwt, path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #[256,256,3]
    img = torch.tensor(image)
    img = img.permute(2,0,1) #[3,256,256]
    img = img.unsqueeze(0)
    subbands = dwt(img)

    LL = subbands[:,0:3,:,:]
    HL = subbands[:,3:6,:,:]
    LH = subbands[:,6:9,:,:]
    HH = subbands[:,9:12,:,:]

    # virtual weight for 4 subbands
    # weights = [0.5,1.0,1.0,1.0]
    # n_LL = LL*weights[0]
    # n_HL = HL*weights[1]
    # n_LH = LH*weights[2]
    # n_HH = HH*weights[3]
    # n_subbands = torch.cat((n_LL, n_HL, n_LH, n_HH), 1)
    # n_Subbands = [torch.permute(n_LL.squeeze_(),(1,2,0))/255.0,torch.permute(n_HL.squeeze_(),(1,2,0))/255.0,torch.permute(n_LH.squeeze_(),(1,2,0))/255.0,torch.permute(n_HH.squeeze_(),(1,2,0))/255.0]
    # show(n_Subbands)

    # show
    Subbands = [torch.permute(LL.squeeze_(),(1,2,0))/255.0,torch.permute(HL.squeeze_(),(1,2,0))/255.0,torch.permute(LH.squeeze_(),(1,2,0))/255.0,torch.permute(HH.squeeze_(),(1,2,0))/255.0]
    show(Subbands)


    reconstruction = idwt(subbands) # [batchsize,c,h,w]
    
    # show
    plt.figure()
    plt.subplot(1,2,1), plt.imshow(torch.permute(reconstruction.squeeze_(),(1,2,0))/255.0), plt.title('reconstruction image'), plt.axis('off')
    plt.subplot(1,2,2), plt.imshow(image/255.0), plt.title('original image'), plt.axis('off')
    plt.show()
    return 

path = r'deepmaterial/utils/dwt/lena.png'
dwt = DWT()
idwt = IWT()
test(dwt,idwt,path)
