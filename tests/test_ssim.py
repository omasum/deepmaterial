import os
from deepmaterial.metrics.psnr_ssim import ssim
import time
import cv2
import torch

def Readimg(img):
    original1 = cv2.imread(img)
    original = cv2.cvtColor(original1, cv2.COLOR_BGR2RGB) # [256, 256, 3]
    original = torch.tensor(original.transpose(2, 0, 1), dtype=torch.float32)
    original = torch.unsqueeze(original, dim=0)
    return original

if __name__ =='__main__':
    starttime= time.time()
    path1 = "tmp/test-parallel.png"
    path2 = "tmp/gimg.png"
    # path2 = "tmp/test-point.png"
    img = Readimg(path1)
    img2 = Readimg(path2)
    value = ssim(img, img2)
    print(value)