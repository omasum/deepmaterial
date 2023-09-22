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
class nafnet(SurfaceNetModel):

    def __init__(self, opt):
        super(nafnet, self).__init__(opt)     

    def feed_data(self, data):
        self.svbrdf = data['svbrdfs'].cuda()
        self.inputs = data['inputs'].cuda()
        self.gt_h = self.HFrequencyGT(self.svbrdf)
        # self.inputs_bands, self.dec = materialmodifier_L6.Show_subbands(self.de_gamma((self.inputs + 1.0)/2.0), Logspace=True)
        # self.inputs_bands = self.inputs_bands[:,0:7,:,:]
        # self.inputs = torch.cat([self.inputs, self.inputs_bands], dim=1).cuda() # [B, 5, H, W]

    def HFrequencyGT(self, svbrdf):
        '''
            turn svbrdf[B, 10, H, W] to svbrdf image dwt decomposition[B, 12, H/2, W/2], range [-1, 1]
        
        '''
        gt_normal = (svbrdf[:,:3] + 1.0) / 2.0 * 255.0 # range(0, 255.0)
        gt_diffuse = (svbrdf[:,3:6] + 1.0) / 2.0 * 255.0 # range(0, 255.0)
        gt_roughness = (svbrdf[:,6:7].repeat(1,3,1,1) + 1.0) / 2.0 * 255.0 # range(0, 255.0)
        gt_specular = (svbrdf[:,7:10] + 1.0) / 2.0 * 255.0 # range(0, 255.0)
        h_normal = self.decomposition(gt_normal)
        h_roughness = self.decomposition(gt_roughness)
        h_specular = self.decomposition(gt_specular)
        h_diffuse = self.decomposition(gt_diffuse)
        gt_h = torch.cat([h_normal, h_diffuse, h_roughness, h_specular], dim=1) #[B, 3*3*4, H/2, W/2]
        return gt_h
    
    def LFrequencyGT(self, svbrdf):
        '''
            turn svbrdf[B, 10, H, W] to svbrdf image dwt decomposition[B, 4*3, H/2, W/2], range [-1, 1]
        
        '''
        gt_normal = self.grayImg((svbrdf[:,:3] + 1.0) / 2.0 * 255.0) # range(0, 255.0)
        gt_diffuse = self.grayImg((svbrdf[:,3:6] + 1.0) / 2.0 * 255.0) # range(0, 255.0)
        gt_roughness = self.grayImg((svbrdf[:,6:7].repeat(1,3,1,1) + 1.0) / 2.0 * 255.0) # range(0, 255.0)
        gt_specular = self.grayImg((svbrdf[:,7:10] + 1.0) / 2.0 * 255.0) # range(0, 255.0)
        h_normal = self.GetLowFrequency(gt_normal)
        h_roughness = self.GetLowFrequency(gt_roughness)
        h_specular = self.GetLowFrequency(gt_specular)
        h_diffuse = self.GetLowFrequency(gt_diffuse)
        gt_h = torch.cat([h_normal, h_diffuse, h_roughness, h_specular], dim=1)
        return gt_h
    
    def GetLowFrequency(self, img):
        '''
            turn gray img [B, 3, H, W] to dwt highfrequency[B, 3, H/2, W/2]
        '''
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        f_inputl, f_inputh = self.dwt(img.cpu())
        f_inputl = torch.round(f_inputl*100000.0)/100000.0
        LowFrequency = f_inputl.cuda()
        # LowFrequency = self.norm(LowFrequency)
        return LowFrequency

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
        self.HighFrequency, output = self.net_g(self.inputs)
        # self.HighFrequency = self.HFrequencyGT(output)
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
        normal, diffuse,roughness,specular = torch.split(output,[3,3,1,3],dim=1)
        normal = torch_norm(normal,dim=1)
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
                HF_save_path = osp.join(self.opt['path']['visualization'], 'HF')
                if self.opt['is_train'] or self.opt['val'].get('save_gt', False):
                    brdf_path=osp.join(save_path, val_data['name'][0])
                    if 'test' in self.opt['datasets'].keys():
                        if self.opt['datasets']['test']['input_mode'] == 'pth':
                            brdf_path = brdf_path[0:-4]
                    self.save_visuals(brdf_path, results['predsvbrdf'], results['gtsvbrdf'])
                    # ######  only test mode print subbands
                    if self.HighFrequency.shape[0] == 1:
                        HF_path = osp.join(HF_save_path, val_data['name'][0])
                        if 'test' in self.opt['datasets'].keys():
                            if self.opt['datasets']['test']['input_mode'] == 'pth':
                                HF_path = HF_path[0:-4]
                        imwrite(tensor2img(self.HighFrequency[0].unsqueeze(1)*0.5+0.5), HF_path)
                else:
                    brdf_path=osp.join(save_path, val_data['name'][0])
                    if 'test' in self.opt['datasets'].keys():
                        if self.opt['datasets']['test']['input_mode'] == 'pth':
                            brdf_path = brdf_path[0:-4]
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
    
    def save_visuals(self, path, pred, gt):
        input = (self.inputs[:,0:3]*0.5+0.5).cpu()
        b,c,h,w = input.shape
        black = torch.zeros(b,c,h,w)
        input_add = torch.cat([input, black], dim=2)
        output = torch.cat([gt,pred],dim=2)*0.5+0.5
        normal, diffuse, roughness, specular = torch.split(output,[3,3,1,3],dim=1)
        roughness = torch.tile(roughness,[1,3,1,1])
        output = torch.cat([normal,diffuse,roughness,specular],dim=-1)
        
        renderer, gtrender = self.eval_render(pred, gt)
        render = torch.split(torch.cat([gtrender,renderer],dim=-2),[1]*self.renderer.nbRendering, dim=-4)
        render = torch.cat(render, dim=-1).squeeze(1)

        output = torch.cat([render**0.4545, output], dim=-1)

        output = torch.cat([input_add, output], dim=3)

        output_img = tensor2img(output,rgb2bgr=True)
        imwrite(output_img, path, float2int=False)

    def save_svbrdfs(self, path, pred):
        
        output = pred*0.5+0.5 # [b, 10, h, w]
        normal, diffuse, roughness, specular = torch.split(output,[3,3,1,3],dim=1)
        roughness = torch.tile(roughness,[1,3,1,1])
        output = torch.cat([normal,diffuse,roughness,specular],dim=-1) # [b, 3, h, 4*w]
        output_img = tensor2img(output,rgb2bgr=True)
        imwrite(output_img, path, float2int=False)

    def SplitFrequency(self, Frequency):
        '''

            Arg:
            Frequency:[B, 12, H/2, W/2] or [B, 4, H/2, W/2]
            Return:
            tensor[nFrequency, dFrequency, rFrequency, sFrequency], [4,B,3,H/2,W/2] or [4, B, 1, H/2, W/2]
        
        '''
        featureFrequency = torch.split(Frequency, split_size_or_sections=int(Frequency.shape[1]/4), dim=1)
        result = torch.cat([featureFrequency[0].unsqueeze_(dim=0), featureFrequency[1].unsqueeze_(dim=0), featureFrequency[2].unsqueeze_(dim=0), featureFrequency[3].unsqueeze_(dim=0)], dim=0)
        return result


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

    # def save_visuals(self, path, pred, gt):
    #     '''
    #         pred. gt [B, 12, H/2, W/2], range[-1, 1]
    #     '''
    #     pred = pred*0.5+0.5 # [B, 12, H/2, W/2], range[0, 1]
    #     gt = gt*0.5+0.5
    #     input = self.HighFrequency*0.5+0.5 # [B, 3, H/2, W/2]
    #     input = self.Split(input)
    #     input = input.repeat(1, 3, 1, 1)

    #     normal, diffuse, roughness, specular = torch.split(pred,[3,3,3,3],dim=1) # list of [B, 3, H/2, W/2]
    #     gt_normal, gt_diffuse, gt_roughness, gt_specular = torch.split(gt,[3,3,3,3],dim=1)
    #     normal = self.Split(normal)
    #     diffuse = self.Split(diffuse)
    #     roughness = self.Split(roughness)
    #     specular = self.Split(specular)
    #     gt_normal = self.Split(gt_normal)
    #     gt_diffuse = self.Split(gt_diffuse)
    #     gt_roughness = self.Split(gt_roughness)
    #     gt_specular = self.Split(gt_specular)

    #     output = torch.cat([normal,diffuse,roughness,specular],dim=-1)
    #     gt_output = torch.cat([gt_normal,gt_diffuse,gt_roughness,gt_specular],dim=-1)
    #     result = torch.cat([output, gt_output], dim=2) # [B, 1, H*2, W*4]
    #     result = result.repeat(1, 3, 1, 1) # [B, 3, H*2, W*4]
    #     original_input = (self.inputs + 1.0)/2.0 # [B, 3, H, W]
    #     original_input = torch.cat([original_input, input], dim=2) # [B, 3, H*2, W]
    #     result = torch.cat([original_input.cpu(), result], dim=3) # [B, 3, H, W*5]
    #     result_img = tensor2img(result,rgb2bgr=True)
    #     imwrite(result_img, path, float2int=False)

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
        rerender = self.renderer.render(pred, n_xy=False, keep_dirs=True, colocated=True)
        gtrender = self.renderer.render(gt, n_xy=False, load_dirs=True, colocated=True)
        return rerender, gtrender
    
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            b, c, h, w = self.inputs.shape
            self.HighFrequency, output = self.net_g(self.inputs)
            normal, diffuse,roughness,specular = torch.split(output,[3,3,1,3],dim=1)
            normal = torch_norm(normal,dim=1)
            self.output = torch.cat([normal, diffuse, roughness, specular],dim=1)
        self.net_g.train()
