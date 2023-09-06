from copy import deepcopy
import os
from time import time
import numpy as np
from torch.utils import data as data
import torch, random

from deepmaterial.data.data_util import paths_from_folder
from deepmaterial.utils import FileClient, imfrombytes, img2tensor, random_tangent\
    , torch_dot, torch_norm, reflect, finit_difference_uv, paraMirror,\
    preprocess, toHDR_torch, toLDR_torch, log_normalization, imresize, Render, PlanarSVBRDF, Lighting
from deepmaterial.utils.registry import DATASET_REGISTRY
from deepmaterial.utils.render_util import PolyRender, RectLighting


@DATASET_REGISTRY.register()
class RealDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read svBRDFs (material for rendering).

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_train (str): Data root path for train svBRDFs.
            dataroot_test (str): Data root path for test svBRDFs.
            io_backend (dict): IO backend type and other kwarg.
            input_type (str): 'brdf' or 'img'
            gt_size (int): Cropped patched size for gt patches.
            split_num (int): number to split read image
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).
            use_ambient (bool): Use ambient light rendering
            jitterlight (bool): Jitter the light and view direction
            
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RealDataset, self).__init__()
        self.opt = opt
        # self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        data_folder = opt['data_path']
        self.input_mode = self.opt.get('input_mode')
        
        self.data_paths = sorted(paths_from_folder(data_folder))
        self.input_path = None

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        # self.sampler = torch.distributions.Uniform(-1.0, 1.0)

    def __getitem__(self, index):

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_path = self.data_paths[index]
        img_bytes = self.file_client.get(img_path, 'brdf')
        img = imfrombytes(img_bytes, float32=True, bgr2rgb=True)
        h, w, c = img.shape
        pattern = {}
        
        if self.input_mode == 'image':
            inputs_img = img2tensor(img, bgr2rgb=False)
            inputs = torch.clip(inputs_img, 0.0, 1.0)

        if not self.opt.get('gamma', True):
            inputs = inputs ** 0.4545
        if self.opt.get('log', False):
            inputs = log_normalization(inputs)
        inputs = preprocess(inputs)

        svbrdfs = torch.zeros(10, h, w)

        result = {
            'inputs': inputs, # input of net, without gamma
            'imgs': inputs_img, # show, with gamma
            'svbrdfs': svbrdfs,
            'name': os.path.basename(img_path)
        }
        result.update(pattern)
        return result
        
    def __len__(self):
        return len(self.data_paths)