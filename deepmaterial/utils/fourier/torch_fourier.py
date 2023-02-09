import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import math

def fft(img):
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

def dfft(m_img):
    ishift = torch.fft.ifftshift(m_img)
    ifft = torch.fft.ifft2(ishift)
    iimg = torch.abs(ifft)
    iimg = iimg.numpy()
    return iimg

# define 10 weights of solid frequency stage
def passfilter(m_img,weight):
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
    filter = np.zeros_like(m_img,dtype='float32')
    for i in range(0,w):
        for j in range(0,h):
            r = math.sqrt(pow(i-w0,2)+pow(j-h0,2))
            for band in bandorigin:
                if band-bandwidth/2<r<band+bandwidth/2:
                    filter[i,j] = weight[bandorigin.index(band)]
                    break
                filter[i,j] = weight[9]

    filter = torch.tensor(filter)
    img = torch.mul(m_img,filter)
    return filter, img



# 读入灰度图,得到三维tuple（h,w,3）
img = cv2.imread(r"D:\cjm\code\fourier\lena.png")
imgr = img[:,:,2]
imgg = img[:,:,1]
imgb = img[:,:,0]
fshift_imgr, magnitude = fft(imgr)
weight = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
filter, imgr_passed = passfilter(fshift_imgr,weight)
new_magnitude = 20*torch.log(torch.abs(imgr_passed))
iimg = dfft(imgr_passed)

sub = imgr-iimg # positive and negtive
plt.subplot(231),plt.imshow(imgr, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(magnitude , cmap = 'gray')
plt.title('spec_img'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(filter , cmap = 'gray')
plt.title('filter'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(new_magnitude , cmap = 'gray')
plt.title('spec_imgr_passed'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(iimg , cmap = 'gray')
plt.title('imgr_passed'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(sub , cmap = 'gray')
plt.title('sub'), plt.xticks([]), plt.yticks([])
plt.savefig(r"D:\cjm\code\fourier\2022passed_png")
plt.show()