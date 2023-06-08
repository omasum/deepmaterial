from collections import OrderedDict
import logging
from deepmaterial.models.surfacenet_model import SurfaceNetModel
from deepmaterial.utils.img_util import imwrite,tensor2img
from deepmaterial.utils.registry import MODEL_REGISTRY
from deepmaterial.utils.render_util import torch_norm
import torch
import importlib
import numpy as np

from deepmaterial.utils.wrapper_util import timmer
metric_module = importlib.import_module('deepmaterial.metrics')
logger = logging.getLogger('deepmaterial')


@MODEL_REGISTRY.register()
class FDModel(SurfaceNetModel):

    def __init__(self, opt):
        super(FDModel, self).__init__(opt)     
                    
    def feed_data(self, data):
        self.svbrdf = data['svbrdfs'].cuda()
        self.inputs = data['inputs'].cuda()

    def debug_rendering(self, gt, pred):
        c,h,w = gt.shape[-3:]
        gt = tensor2img(gt.view(-1, c, h, w), gamma=True)
        pred =tensor2img(pred.view(-1, c, h, w), gamma=True)
        output_img = np.concatenate([gt, pred], axis=0)
        imwrite(output_img, 'tmp/debug_rendering.png', float2int=False)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        if current_iter==1:
            logger.warning('Train all the parameters.')
            for p in self.net_g.parameters():
                p.requires_grad = True
        if self.opt.get('network_d'):
            # optimize net_g
            for p in self.net_d.parameters():
                p.requires_grad = False
        
        self.optimizer_g.zero_grad()
        output = self.net_g(self.inputs)
        l_total = self.computeLoss(output, loss_dict)
        # if torch.isnan(l_total):
        #     print("wandan")
        l_total.backward()
        self.optimizer_g.step()
        if current_iter % self.opt_d_every == 0 and self.cri_gan is not None:
            # # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True
                
            self.optimizer_d.zero_grad()

            l_d = self.computeLoss(output, loss_dict, isDisc=True)
            
            l_d.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def computeLoss(self, output, loss_dict, isDesc=False):
        normal, diffuse,roughness,specular = torch.split(output,[3,3,3,3],dim=1)
        roughness = torch.mean(roughness, dim=1, keepdim=True)
        return self.brdfLoss(normal, diffuse, roughness, specular, loss_dict, isDesc)
    
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            b, c, h, w = self.inputs.shape
            output = self.net_g(self.inputs)
            normal, diffuse,roughness,specular = torch.split(output,[3,3,3,3],dim=1)
            roughness = torch.mean(roughness, dim=1, keepdim=True)
            self.output = torch.cat([normal, diffuse, roughness, specular],dim=1)
        self.net_g.train()
