import numpy as np
import cv2
import os
from deepmaterial.archs.vgg_wfc_sf_test_arch import VGG_wfc_sf_test
import torch

original_path = "/home/sda/svBRDFs/testBlended/0000002;PolishedMarbleFloor_01Xmetal_bumpy_squares;3Xdefault.png"
path = "results/001_RADN_archived_20230612_152326/visualization/areaDataset/svbrdf-001_RADN-1.png"
target = "tests/Fimg_001_RADN_archived_20230405_180124"

allimg = cv2.imread(path)
allimg = cv2.cvtColor(allimg, cv2.COLOR_BGR2RGB)
[original, pred, filter] = np.split(allimg,indices_or_sections=3, axis=0)

original1 = cv2.imread(original_path)
original = cv2.cvtColor(original1, cv2.COLOR_BGR2RGB)

input = np.split(original, 5, axis=1)[0]
input = torch.tensor(input.transpose(2, 0, 1)) #torch.tensor(3,256,256)
filters = np.split(filter/255.0, 5, axis=1) # [black, filtern, filterd, filterr, filters]
if filters[0].max()!=0.0:
    print("error") 

vgg = VGG_wfc_sf_test('VGG11')
f_input = vgg.fft(input)

def passfilter(m_img,filter):
    '''passfilter for frequency domain image

    Args:
        m_img (tensor.complex[batchsize,3,h,w]): after fft image
        filter (tensor.float[batchsize,3,h,w]): filter
    Returns:
        tensor.complex[batchsize,3,h,w]: new image
    '''

    img = torch.mul(m_img,filter)
    new_magnitude = 20*torch.log(torch.abs(img))
    new_magnitude = torch.clip(new_magnitude, min=0.0, max=255.0)
    return img, new_magnitude
    # img corresponds to complex value

def dfft(m_img):
    '''dfft of passfiltered images

    Args:
        m_img (tensor.complex[batchsize,3,h,w]): frequency domain images

    Returns:
        iimg(tensor.float[batchsize,3,h,w]): images
    '''
    ishift = torch.fft.ifftshift(m_img)
    ifft = torch.fft.ifft2(ishift)
    iimg = torch.abs(ifft)
    rimg = torch.real(ifft)
    iimg = torch.clip(iimg, min=0.0, max=255.0)
    rimg = torch.clip(rimg, min=0.0, max=255.0)
    return iimg, rimg

n, nf = passfilter(f_input, torch.tensor(filters[1].transpose(2, 0, 1)))
d, df = passfilter(f_input, torch.tensor(filters[2].transpose(2, 0, 1)))
r, rf = passfilter(f_input, torch.tensor(filters[3].transpose(2, 0, 1)))
s, sf = passfilter(f_input, torch.tensor(filters[4].transpose(2, 0, 1)))

n, nr = dfft(n)
d, dr = dfft(d)
r, rr = dfft(r)
s, sr = dfft(s)

print("n diff:" ,abs(df-nf).mean())
print("d diff:" ,abs(rf-df).mean())
print("r diff:" ,abs(sf-rf).mean())
print("s diff:" ,abs(sf-nf).mean())

result = (torch.cat([n,d,r,s], dim = 2)).numpy().transpose(1, 2, 0).astype(np.uint8)
f_result = (torch.cat([nf,df,rf,sf], dim = 2)).numpy().transpose(1, 2, 0).astype(np.uint8)
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
f_result = cv2.cvtColor(f_result, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(target,'filtered_img.png'), result)
cv2.imwrite(os.path.join(target,'f_filtered_img.png'), f_result)
