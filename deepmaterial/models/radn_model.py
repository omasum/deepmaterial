from collections import OrderedDict
import logging
from deepmaterial.models.surfacenet_model import SurfaceNetModel
from deepmaterial.utils.img_util import imwrite,tensor2img
from deepmaterial.utils.registry import MODEL_REGISTRY
from deepmaterial.utils.render_util import torch_norm
import torch
import importlib
import numpy as np
import os.path as osp
from tqdm import tqdm
from copy import deepcopy

from deepmaterial.utils.wrapper_util import timmer
metric_module = importlib.import_module('deepmaterial.metrics')
logger = logging.getLogger('deepmaterial')


@MODEL_REGISTRY.register()
class RADNModel(SurfaceNetModel):

    def __init__(self, opt):
        super(RADNModel, self).__init__(opt)     

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
        normal, diffuse,roughness,specular = torch.split(output,[2,3,1,3],dim=1)
        normal = torch_norm(torch.cat([normal,torch.ones_like(roughness, device=self.device)], dim=1),dim=1)
        return self.brdfLoss(normal, diffuse, roughness, specular, loss_dict, isDesc)

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        if self.opt.get('pbar',True):
            pbar = tqdm(total=len(dataloader), unit='image')
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            if self.opt['val'].get('save_img', False):
                results = self.get_current_visuals(self.output.cpu(), self.svbrdf.cpu())
                save_path = osp.join(self.opt['path']['visualization'], dataset_name)
                if self.opt['is_train'] or self.opt['val'].get('save_gt', False):
                    brdf_path=osp.join(save_path, f'svbrdf-{current_iter}-{idx}.png')
                    self.save_visuals(brdf_path, results['predsvbrdf'], results['gtsvbrdf'])
                else:
                    brdf_path=osp.join(save_path, val_data['name'][0])
                    self.save_pre_visuals(brdf_path,results['predsvbrdf'])
                    
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                # print('pred:',self.output)
                # print('gt:',self.brdf)
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    if metric_type == 'cos':
                        error = self.cri_cos(self.output, self.svbrdf)*opt_.pop('weight')
                    elif metric_type == 'pix':
                        error = torch.abs(self.output-self.svbrdf).mean()*opt_.pop('weight')
                    self.metric_results[name] += error 
            torch.cuda.empty_cache()
            if self.opt.get('pbar',True):
                pbar.update(1)
                pbar.set_description(f'Testing')
                # break
        if self.opt.get('pbar',True):
            pbar.close()
            
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name,
                                                tb_logger)

    def save_visuals(self, path, pred, gt):
        output = torch.cat([gt,pred],dim=2)*0.5+0.5
        normal, diffuse, roughness, specular = torch.split(output,[3,3,1,3],dim=1)
        roughness = torch.tile(roughness,[1,3,1,1])
        output = torch.cat([normal,diffuse,roughness,specular],dim=-1)
        
        renderer, gtrender = self.eval_render(pred, gt)
        render = torch.split(torch.cat([gtrender,renderer],dim=-2),[1]*self.renderer.nbRendering, dim=-4)
        render = torch.cat(render, dim=-1).squeeze(1)

        output = torch.cat([render**0.4545, output], dim=-1)

        output_img = tensor2img(output,rgb2bgr=True)
        imwrite(output_img, path, float2int=False)

    def eval_render(self, pred, gt):
        rerender = self.renderer.render(pred, n_xy=False, keep_dirs=True, light_dir = torch.tensor([0, 0.3, 1]))
        gtrender = self.renderer.render(gt, n_xy=False, load_dirs=True)
        return rerender, gtrender
    
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            b, c, h, w = self.inputs.shape
            output = self.net_g(self.inputs)
            normal, diffuse,roughness,specular = torch.split(output,[2,3,1,3],dim=1)
            normal = torch_norm(torch.cat([normal,torch.ones((b,1,h,w), device=self.device)], dim=1),dim=1)
            self.output = torch.cat([normal, diffuse, roughness, specular],dim=1)
        self.net_g.train()
