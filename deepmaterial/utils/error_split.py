import numpy as np
import cv2
import os
import torch

original_path = "results/U_NET1v1/visualization/areaDataset"

def error():
    error = 0.0
    nerror = 0.0
    derror = 0.0
    rerror = 0.0
    serror = 0.0
    for file in os.listdir(original_path):

        original1 = cv2.imread(os.path.join(original_path, file))
        original = cv2.cvtColor(original1, cv2.COLOR_BGR2RGB) # [256*2, 256*5, 3]
        original = torch.tensor(original.transpose(2, 0, 1), dtype=torch.float32) # torch.tensor(3, 256*2, 256*5)
        gt1, pred1 = torch.split(original/255.0, split_size_or_sections=int(original.shape[1]/2), dim=1)
        all_gt = torch.split(gt1, int(gt1.shape[2]/5), dim=2)
        all_preds = torch.split(pred1, int(pred1.shape[2]/5), dim=2)
        gt, gtn, gtd, gtr, gts = Getvalue(all_gt)
        pred, predn, predd, predr, preds = Getvalue(all_preds)
        error += torch.abs(gt - pred).mean()
        nerror += torch.abs(gtn - predn).mean()
        derror += torch.abs(gtd - predd).mean()
        rerror += torch.abs(gtr - predr).mean()
        serror += torch.abs(gts - preds).mean()
    print("error: ", error/84)
    print("nerror: ", nerror/84)
    print("derror: ", derror/84)
    print("rerror: ", rerror/84)
    print("serror: ", serror/84)


def Getvalue(imgs):
    f_l = []
    for i in range(0,5):
        input = imgs[i].unsqueeze(0) # torch.tensor(1, C, 256, 256)

        f_l.append(input)
    return f_l

error()