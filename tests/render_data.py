import math
from typing import List
import numpy as np
import torch
from deepmaterial.utils.ltc_util import LTC
from deepmaterial.utils.vector_util import numpy_norm, torch_cross, torch_norm, torch_dot
from deepmaterial.utils.img_util import gaussianBlurConv, img2tensor, toLDR_torch, toHDR_torch, preprocess
from abc import ABCMeta, abstractmethod
import cv2 as cv
from torch.nn import functional as F
from deepmaterial.utils.wrapper_util import timmer
import nvdiffrast.torch as dr
from deepmaterial.utils.render_util import PlanarSVBRDF, Render
import cv2
import time
import os
from deepmaterial.utils.img_util import imwrite,tensor2img

lightDistance = 2.14
# lightDistance = 1.14
viewDistance = 5  # 44 degrees FOV 

brdfArgs = {}
brdfArgs['nbRendering'] = 1
brdfArgs['size'] = 256
brdfArgs['order'] = 'pndrs'
brdfArgs['toLDR'] = True
brdfArgs['lampIntensity'] = 12
paths = '/home/cjm/dataset/all_test/GT' # [256, 256*5]indrs
target = "/home/cjm/dataset/rerender_test"
svbrdf = PlanarSVBRDF(brdfArgs)

for name in os.listdir(paths):

    img = cv2.imread(os.path.join(paths, name))[:, :, ::-1] / 255.0 # bgr2rgb
    svbrdfs = svbrdf.get_svbrdfs(img) # change svbrdf.brdf
    svbrdfs = torch.clip(svbrdfs, -1.0, 1.0)
    # svbrdf.brdf[:3,:,:] = torch.stack([torch.zeros_like(svbrdf.brdf[0,:,:]),torch.zeros_like(svbrdf.brdf[0,:,:]),torch.ones_like(svbrdf.brdf[0,:,:])])
    # svbrdf.brdf[3:6,:,:] = torch.ones_like(svbrdf.brdf[0:1,:,:])*(-0.0)
    # svbrdf.brdf[6:7,:,:] = torch.ones_like(svbrdf.brdf[0:1,:,:])*(-0.8)
    # svbrdf.brdf[7:10,:,:] = torch.ones_like(svbrdf.brdf[0:1,:,:])*(1.0)
    
    #====================== Rendering test============================
    renderer = Render(brdfArgs)
    start = time.time()
    res = renderer.render(svbrdfs, random_light=True, colocated=True) # [0,1]
    print('point rendering:', time.time()-start)
    svbrdfs = (svbrdfs+1.0)/2.0
    normal, diffuse, roughness, specular = torch.split(svbrdfs,[3,3,1,3],dim=0)
    roughness = torch.tile(roughness,[3,1,1])
    svbrdfs = torch.cat([normal,diffuse,roughness,specular],dim=-1)
    result = torch.cat([res, svbrdfs], dim=-1)
    result = tensor2img(result,rgb2bgr=True)
    imwrite(result, os.path.join('/home/cjm/dataset/rerender_test', name), float2int=False)