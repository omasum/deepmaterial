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
import cv2
from pytorch_wavelets import DWTForward, DWTInverse
from deepmaterial.metrics.psnr_ssim import ssim

from deepmaterial.utils.wrapper_util import timmer
from deepmaterial.utils.materialmodifier import materialmodifier_L6

metric_module = importlib.import_module('deepmaterial.metrics')
logger = logging.getLogger('deepmaterial')


@MODEL_REGISTRY.register()
class Refine_roughness(SurfaceNetModel):

    def __init__(self, opt):
        super(Refine_roughness, self).__init__(opt)     

    def feed_data(self, data):
        self.svbrdf = data['svbrdfs'].cuda() # [b,3,h,w], gt normal
        self.svbrdf = torch.mean(self.svbrdf, dim=1, keepdim=True) # [b,1,h,w]
        self.inputs = data['inputs'].cuda() # [b,3,h,w], pred normal
        # self.gt_h = self.HFrequencyGT(self.svbrdf)
        self.inputs_bands, self.dec = materialmodifier_L6.Show_subbands(self.de_gamma((self.inputs + 1.0)/2.0), Logspace=True)
        self.inputs_bands = self.inputs_bands[:,4:5,:,:] # the second band of choosen
        self.inputs = torch.cat([self.inputs, self.inputs_bands], dim=1).cuda() # [B, 3+8, H, W]


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
        b, c, h, w = output.shape
        return self.brdfLoss(output, loss_dict, isDesc)

    def brdfLoss(self, normal, loss_dict, isDesc=False):
        l_total = 0
        if not isDesc:
            gt = self.svbrdf
            l_pix = self.cri_pix(normal, gt)
            loss_dict['l_pix'] = l_pix
            l_total += l_pix
            return l_total 
    

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

                    elif metric_type == 'HF':
                        error = torch.abs(self.HFrequencyGT(self.output)-self.HFrequencyGT(self.svbrdf)).mean()*opt_.pop('weight')
                    
                    elif metric_type == 'LF':
                        error = torch.abs(self.LFrequencyGT(self.output)-self.LFrequencyGT(self.svbrdf)).mean()*opt_.pop('weight')

                    elif metric_type == 'pix':
                        error = torch.abs(self.output-self.svbrdf).mean()*opt_.pop('weight')
                    elif metric_type == 'normal_pix':
                        error = torch.abs(self.output[:, :3] - self.svbrdf[:, :3]).mean()*opt_.pop('weight')
                    elif metric_type == 'diffuse_pix':
                        error = torch.abs(self.output[:, 3:6] - self.svbrdf[:, 3:6]).mean()*opt_.pop('weight')
                    elif metric_type == 'roughness_pix':
                        error = torch.abs(self.output[:, 6:7] - self.svbrdf[:, 6:7]).mean()*opt_.pop('weight')
                    elif metric_type == 'specular_pix':
                        error = torch.abs(self.output[:, 7:10] - self.svbrdf[:, 7:10]).mean()*opt_.pop('weight')
                    elif metric_type == 'ssim':
                        error = torch.abs(ssim(self.deprocess(self.output), self.deprocess(self.svbrdf)))*opt_.pop('weight')
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
    

    def deprocess(self,m_img):
        '''depreocess images fed to network(-1,1) to initial images(0,255)

        Args:
            m_img (tensor.complex[batchsize,3,h,w]): images fetched
        '''
        # (-1~1) convert to (0~1)
        img = (m_img+1.0)/2.0

        #de-log_normalization
        img = self.de_gamma(img)

        #(0,1) convert to (0,255)
        img = img * 255.0
        return img
    
    def de_gamma(self,img):
        image = img**2.2
        image = torch.clip(image, min=0.0, max=1.0)
        return image

    def save_visuals(self, path, pred, gt):
        output = torch.cat([gt,pred],dim=2)*0.5+0.5 # [b,3,2*h,w]
        output = output.repeat(1,3,1,1)
        output_img = tensor2img(output,rgb2bgr=True)
        imwrite(output_img, path, float2int=False)

    def Split(self, img):
        '''
            compose subband-images to 1 image
            Args:
            img: tensor[B, C*3, H/2**j, W/2**j]
            Return:
            image: tensor[B, C, H/2**(j-1), W/2**(j-1)]
        '''
        result = torch.split(img, split_size_or_sections=int(img.shape[1]/3),dim=1)
        black = torch.zeros_like(result[0])
        imglin1 = torch.cat([black, result[0]], dim=3)
        imglin2 = torch.cat([result[1], result[2]], dim=3)
        img = torch.cat([imglin1, imglin2], dim=2)
        return img

    def eval_render(self, pred, gt):
        # rerender = self.renderer.render(pred, n_xy=False, keep_dirs=True, light_dir = torch.tensor([0, 0.3, 1]))
        rerender = self.renderer.render(pred, n_xy=False, keep_dirs=True)
        gtrender = self.renderer.render(gt, n_xy=False, load_dirs=True)
        return rerender, gtrender
    
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            b, c, h, w = self.inputs.shape
            output = self.net_g(self.inputs)
            self.output = output
        self.net_g.train()
