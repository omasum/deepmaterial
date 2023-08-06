import numpy as np
import cv2
import os
import torch
from pytorch_wavelets import DWTForward, DWTInverse

level = 1
dwt = DWTForward(J=level, wave='db2', mode='zero')
idwt = DWTInverse(wave='db2', mode='zero')

original_path = "results/U_NET_L1Loss/visualization/areaDataset"

def Analyze():
    lerror = 0.0
    herror = 0.0
    lnerror = 0.0
    lderror = 0.0
    lrerror = 0.0
    lserror = 0.0
    hnerror = 0.0
    hderror = 0.0
    hrerror = 0.0
    hserror = 0.0
    for file in os.listdir(original_path):

        original1 = cv2.imread(os.path.join(original_path, file))
        original = cv2.cvtColor(original1, cv2.COLOR_BGR2RGB) # [256*2, 256*5, 3]

        original = torch.tensor(original.transpose(2, 0, 1), dtype=torch.float32) # torch.tensor(3, 256*2, 256*5)
        # original = torch.unsqueeze(1/3*original[0,:,:] + 1/3*original[1,:,:] + 1/3*original[2,:,:], dim=0) #[1,h,w]
        gt, pred = torch.split(original, split_size_or_sections=int(original.shape[1]/2), dim=1)
        gts = torch.split(gt, int(gt.shape[2]/5), dim=2)
        preds = torch.split(pred, int(pred.shape[2]/5), dim=2)
        gtl, gth = GetFrequency(gts)
        predl, predh = GetFrequency(preds)
        lnerror += torch.abs(gtl[0] - predl[0]).mean()
        hnerror += torch.abs(gth[0] - predh[0]).mean()
        lderror += torch.abs(gtl[1] - predl[1]).mean()
        hderror += torch.abs(gth[1] - predh[1]).mean()
        lrerror += torch.abs(gtl[2] - predl[2]).mean()
        hrerror += torch.abs(gth[2] - predh[2]).mean()
        lserror += torch.abs(gtl[3] - predl[3]).mean()
        hserror += torch.abs(gth[3] - predh[3]).mean()
        lerror += torch.cat([torch.abs(gtl[0] - predl[0]), torch.abs(gtl[1] - predl[1]), torch.abs(gtl[2] - predl[2]), torch.abs(gtl[3] - predl[3])], dim=3).mean()
        herror += torch.cat([torch.abs(gth[0] - predh[0]), torch.abs(gth[1] - predh[1]), torch.abs(gth[2] - predh[2]), torch.abs(gth[3] - predh[3])], dim=3).mean()
    print("low error: ", lerror/84.0)
    print("low nerror: ", lnerror/84.0)
    print("low derror: ", lderror/84.0)
    print("low rerror: ", lrerror/84.0)
    print("low serror: ", lserror/84.0)
    print("highfrequency error: ", herror/84.0)
    print("highfrequency nerror: ", hnerror/84.0)
    print("highfrequency derror: ", hderror/84.0)
    print("highfrequency rerror: ", hrerror/84.0)
    print("highfrequency serror: ", hserror/84.0)

def GetFrequency(imgs):
    f_l = []
    f_h = []
    for i in range(1,5):
        input = imgs[i].unsqueeze(0) # torch.tensor(1, C, 256, 256)

        f_inputl, f_inputh = dwt(input) # f_inputl: tensor(B,C,H/2**level,W/2**level), f_inputh[i]: tensor(B,C,3,H/2**(i+1),W/2**(i+1))

        # r_input = idwt((f_inputl, f_inputh)) # idwt from dwt image
        f_l.append(f_inputl)
        f_h.append(f_inputh[0])
    return f_l, f_h
    
Analyze()