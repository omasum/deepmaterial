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

paths = 'results/NAFNetHFRenderLoss_allbands_render_allset/visualization/areaDataset' # [256, 256*5]indrs
target = "/home/cjm/dataset/rerender_test"

for name in os.listdir(paths):

    img = cv2.imread(os.path.join(paths, name)) # bgr, [h,w,c]
    gt = np.split(img, 2, 0)[0]
    input, rerender, n, d, r, s = np.split(gt, 6, 1)
    result = np.concatenate([input, n, d, r, s], axis=1)
    cv2.imwrite(os.path.join(target, name), result)