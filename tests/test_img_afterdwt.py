import numpy as np
import cv2
import os
import torch





def dwt(img):
    '''Haar DWT of image

    Args:
        img (tensor.float[batchsize,3,h,w]): original images

    Returns:
        dwt_img(tensor.float[batchsize,c*4,h/2,w/2]):return cat of 4 feature maps
    '''
    
    x01 = img[:, 0::2, :] / 2 # odd row
    x02 = img[:, 1::2, :] / 2 # even row
    x1 = x01[:, :, 0::2] # odd row, odd column
    x2 = x02[:, :, 0::2] # even row, odd column
    x3 = x01[:, :, 1::2] # odd row, even column
    x4 = x02[:, :, 1::2] # even row, even column
    x_LL = x1 + x2 + x3 + x4 # height/2, width/2
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

def passfilter(m_img,filter):
    '''passfilter for frequency domain image

    Args:
        m_img (tensor.float[batchsize, 4*3, h/2, w/2]): after dwt image
        filter (tensor.float[batchsize, 4*3, h/2, w/2]): filter for LL, HL, LH, HH features
    Returns:
        tensor.float[batchsize,4*3,h/2,w/2]: new wavelet features of each svbrdf maps
    '''

    img = torch.mul(m_img,filter)
    # new_magnitude = 20*torch.log(torch.abs(img))
    
    return img
    # img corresponds to complex value

def idwt(m_img):
    ''' Haar iDWT of features

    Args:
        m_img (tensor.float[batchsize,c*4,h/2,w/2]): wavelet features

    Returns:
        iimg(tensor.float[batchsize,3,h,w]): images
    '''
    r = 2
    in_channel, in_height, in_width = m_img.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_channel, out_height, out_width = int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = m_img[0:out_channel, :, :] / 2
    x2 = m_img[out_channel:out_channel * 2, :, :] / 2
    x3 = m_img[out_channel * 2:out_channel * 3, :, :] / 2
    x4 = m_img[out_channel * 3:out_channel * 4, :, :] / 2
    

    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_channel, out_height, out_width]).float()

    h[:, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

def FilterSplit(filter):
    '''filter [h, w, 3] to [h/2, w/2, 3*4]

    Args:
        fitler (_type_): _description_
    '''
    filterline = np.split(filter, indices_or_sections=2, axis=0)
    filter1 = np.split(filterline[0], indices_or_sections=2, axis=1)
    filter2 = np.split(filterline[1], indices_or_sections=2, axis=1)
    filter = np.concatenate((filter1[0], filter1[1], filter2[0], filter2[1]), axis=2)
    return filter

def Split(img):

    result = torch.split(img, split_size_or_sections=int(img.shape[0]/4),dim=0)
    imglin1 = torch.cat([result[0], result[1]], dim=2)
    imglin2 = torch.cat([result[2], result[3]], dim=2)
    img = torch.cat([imglin1, imglin2], dim=1)
    return img


original_path = r"/home/sda/svBRDFs/testBlended"
path = r"D:\cjm\code\my project\fourier\DataAnalysis\RADN\wavelet\svbrdf-001_RADN-8.png"
target = r"D:\cjm\code\my project\fourier\DataAnalysis\RADN\wavelet"

black = torch.zeros(3, 64, 64)
white = torch.ones(3, 64, 64)
filter = []
filter[0] = torch.cat([white, black, black, black], dim=0)
filter[1] = torch.cat([black, white, black, black], dim=0)
filter[2] = torch.cat([black, black, white, black], dim=0)
filter[3] = torch.cat([black, black, black, white], dim=0)

for file in os.listdir(original_path):

    original1 = cv2.imread(os.path.join(original_path, file))
    original = cv2.cvtColor(original1, cv2.COLOR_BGR2RGB) # [256, 256*5, 3]

    original = torch.tensor(original.transpose(2, 0, 1)) # torch.tensor(3, 256, 256*5)
    original = torch.unsqueeze(1/3*original[0,:,:] + 1/3*original[1,:,:] + 1/3*original[2,:,:], dim=0) #[1,h,w]
    input = torch.split(original, int(original.shape[2]/5), dim=2)[0]


    f_input = dwt(input)

    n = passfilter(f_input, filter[0])
    d = passfilter(f_input, filter[1])
    r = passfilter(f_input, filter[2])
    s = passfilter(f_input, filter[3])


    result = (torch.cat([Split(n),Split(d),Split(r),Split(s)], dim = 2)).numpy().transpose(1, 2, 0).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    f_result = cv2.cvtColor(f_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(target,'filtered_img.png'), result)
    cv2.imwrite(os.path.join(target,'f_filtered_img.png'), f_result)
