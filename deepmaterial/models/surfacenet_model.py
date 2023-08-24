from collections import OrderedDict
from copy import deepcopy
import logging
from tqdm import tqdm
from deepmaterial.archs import build_network
from deepmaterial.losses import build_loss
from deepmaterial.utils.img_util import imwrite,tensor2img
from deepmaterial.utils.logger import get_root_logger
from deepmaterial.utils.registry import MODEL_REGISTRY
from deepmaterial.utils.render_util import Render,torch_norm
from .base_model import BaseModel
import torch
import importlib
import os.path as osp
from pytorch_wavelets import DWTForward, DWTInverse
metric_module = importlib.import_module('deepmaterial.metrics')
logger = logging.getLogger('deepmaterial')


@MODEL_REGISTRY.register()
class SurfaceNetModel(BaseModel):
    def __init__(self, opt):
        super(SurfaceNetModel, self).__init__(opt)
        # define network
        self.brdf_args = opt['network_g'].pop('brdf_args')
        self.initRender()
        self.initNetworks(opt.get('print_net',True))
        if self.is_train:
            self.init_training_settings()
            self.opt_d_every = opt['train'].get('opt_d_every', 1)

    def initRender(self):
        self.renderer = Render(self.brdf_args)
        self.surface = self.renderer.generate_surface(self.renderer.size).to(self.device)

    def initNetworks(self, printNet=True):
        self.net_g = self.buildNet(self.opt.get('network_g'), 'g', printNet)
        if self.opt.get('network_d'):
            self.net_d = self.buildNet(self.opt.get('network_d'), 'd', printNet)

    def buildNet(self, net_opt, path_key, printNet=True):
        net = build_network(net_opt)
        net = self.model_to_device(net)
            
        if printNet:
            self.print_network(net)

        load_path = self.opt['path'].get('pretrain_network_' + path_key, None)
        # load pretrained models
        if load_path is not None:
            params = 'params'
            self.load_network(net, load_path, self.opt['path'].get('strict_load_'+path_key, True), params)
        return net

    def init_training_settings(self):
        self.net_g.train()
        
        if self.opt.get('network_d'):
            self.net_d.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        # if train_opt.get('single_opt'):
        #     self.cri_singlepix = build_loss(train_opt['single_opt']).to(self.device)
        # else:
        #     self.cri_singlepix = None
        if train_opt.get('render_opt'):
            self.cri_render = build_loss(train_opt['render_opt']).to(self.device)
        else:
            self.cri_render = None
        # add frequency loss
        if train_opt.get('highfrequency_opt'):
            self.cri_highfrequency = build_loss(train_opt['highfrequency_opt']).to(self.device)
        else:
            self.cri_highfrequency = None
        if train_opt.get('noramalpixel_opt'):
            self.cri_noramalpix = build_loss(train_opt['noramalpixel_opt']).to(self.device)
        else:
            self.cri_noramalpix = None
        if train_opt.get('diffusepixel_opt'):
            self.cri_diffusepix = build_loss(train_opt['diffusepixel_opt']).to(self.device)
        else:
            self.cri_diffusepix = None
        if train_opt.get('roughnesspixel_opt'):
            self.cri_roughnesspix = build_loss(train_opt['roughnesspixel_opt']).to(self.device)
        else:
            self.cri_roughnesspix = None
        if train_opt.get('specularpixel_opt'):
            self.cri_specularpix = build_loss(train_opt['specularpixel_opt']).to(self.device)
        else:
            self.cri_specularpix = None
        if train_opt.get('msssim_opt'):
            self.cri_msssim = build_loss(train_opt['msssim_opt']).to(self.device)
        else:
            self.cri_msssim = None
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # set g
        optim_params = [param for param in self.net_g.parameters() if param.requires_grad]
        for name,_ in self.net_g.named_parameters():
            print(name)
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # set d
        if self.opt.get('network_d'):
            optim_params = [param for param in self.net_d.parameters() if param.requires_grad]
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, optim_params, **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)
        
    def feed_data(self, data):
        self.svbrdf = data['svbrdfs'].cuda()
        self.inputs = data['inputs'].cuda()

    def eval_render_train(self, pred, gt):
        b, _, h,w = pred.shape
        tmp = self.renderer.nbRendering
        self.renderer.nbRendering = self.opt['train'].get('n_diff', 3) + self.opt['train'].get('n_sepc', 6)
        wi, wo = self.renderer.fullRandomCosine(b, 0, 6)
        l, v, d, p = self.renderer.torch_generate(wo.to(self.device), wi.to(self.device), surface=self.surface)
        wi, wo = self.renderer.fullRandomCosine(b, 3, 0)
        l, v = torch.cat([wi.view(b, 3, 3,1,1).to(self.device).repeat(1,1,1,h,w), l], dim=-4),\
            torch.cat([wo.view(b, 3,3,1,1).to(self.device).repeat(1, 1,1,h,w), v], dim=-4)
        d = torch.ones_like(l)
        
        self.renderer.lighting.set_lamp_intensity(torch.pi)
        gt_render = self.renderer.render(gt, light_dir=l, view_dir=v, surface=p, light_dis=d, n_xy=False, toLDR=False, lossRender=True)
        gt_render /= torch.clip(l[...,2:,:,:], 0.001)
        pred_render = self.renderer.render(pred, light_dir=l, view_dir=v, surface=p, light_dis=d, n_xy=False, toLDR=False, lossRender=True)
        pred_render /= torch.clip(l[...,2:,:,:], 0.001)
        self.renderer.lighting.reset_lamp_intensity()
        self.renderer.nbRendering = tmp
        
        return gt_render, pred_render
    
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
        fake_img = self.net_g(self.inputs)
        normal, diffuse,roughness,specular = torch.split(fake_img,list(self.opt['network_g']['outputMapping'].values()),dim=1)
        normal = torch_norm(normal,dim=1)

        l_total = self.brdfLoss(normal, diffuse, roughness, specular, loss_dict)

        l_total.backward()
        self.optimizer_g.step()
        if current_iter % self.opt_d_every == 0 and self.cri_gan is not None:
            # # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True
                
            self.optimizer_d.zero_grad()
            l_d = self.brdfLoss(normal, diffuse, roughness, specular, loss_dict, True)
            l_d.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def brdfLoss(self, normal, diffuse, roughness, specular, loss_dict, isDesc=False):
        if self.cri_gan is not None:
            fake_img = torch.cat([self.inputs,normal,diffuse,roughness,specular],dim=1)
        l_total = 0
        if not isDesc:
            if self.cri_render is not None:
                gt_render, pred_render = self.eval_render_train(torch.cat([normal, diffuse, roughness, specular], dim=1), self.svbrdf)
                # self.debug_rendering(gt_render.detach().cpu(), pred_render.detach().cpu())
                l_render = self.cri_render(gt_render, pred_render)
                # del gt_render, pred_render
                loss_dict['l_render'] = l_render
                l_total += l_render 
            if self.cri_pix is not None:
                r = torch.cat([roughness]*3, dim=1)
                output = torch.cat([normal, diffuse, r, specular], dim=1)
                gt = torch.cat([self.svbrdf[:,:6], self.svbrdf[:,6:7], self.svbrdf[:,6:7], self.svbrdf[:,6:7], self.svbrdf[:,7:10]], dim=1)
                l_pix = self.cri_pix(output, gt)
                loss_dict['l_pix'] = l_pix
                l_total += l_pix
            # if self.cri_singlepix is not None:
            #     r = torch.cat([roughness]*3, dim=1)
            #     output = torch.cat([normal, diffuse, r, specular], dim=1)
            #     gt = torch.cat([self.svbrdf[:,:6], self.svbrdf[:,6:7], self.svbrdf[:,6:7], self.svbrdf[:,6:7], self.svbrdf[:,7:10]], dim=1)
            #     l_pix = self.cri_pix(output, gt)
            #     loss_dict['l_pix'] = l_pix
            #     l_total += l_pix
            if self.cri_msssim is not None:
                r = torch.cat([roughness]*3, dim=1)
                ls_n = self.cri_msssim(normal, self.svbrdf[:,0:3])
                ls_a = self.cri_msssim(diffuse, self.svbrdf[:,3:6])
                ls_r = self.cri_msssim(roughness, gt[:,6:9])
                ls_s = self.cri_msssim(specular, self.svbrdf[:,9:12])
                l_msssim = ls_n+ls_a+ls_r+ls_s
                loss_dict['l_msssim'] = l_msssim
                l_total += l_msssim
            if self.cri_gan is not None:
                fake_pred = self.net_d(fake_img)
                l_g = self.cri_gan(fake_pred,True,is_disc=False)
                loss_dict['l_g'] = l_g
                l_total += l_g
            if self.cri_highfrequency is not None:
                r = torch.cat([roughness]*3, dim=1)
                output = torch.cat([normal, diffuse, r, specular], dim=1)
                l_pix = self.cri_highfrequency(self.HFrequency(output), self.gt_h)
                loss_dict['l_frequencypix'] = l_pix
                l_total += l_pix
            if self.cri_noramalpix is not None:
                output = normal
                gt = self.svbrdf[:,:3]
                normal_pix = self.cri_pix(output, gt)
                loss_dict['normal_pix'] = normal_pix
            if self.cri_diffusepix is not None:
                output = diffuse
                gt = self.svbrdf[:,3:6]
                diffuse_pix = self.cri_pix(output, gt)
                loss_dict['diffuse_pix'] = diffuse_pix
            if self.cri_roughnesspix is not None:
                output = roughness
                gt = self.svbrdf[:,6:7]
                roughness_pix = self.cri_pix(output, gt)
                loss_dict['roughness_pix'] = roughness_pix
            if self.cri_specularpix is not None:
                output = specular
                gt = self.svbrdf[:,7:10]
                specular_pix = self.cri_pix(output, gt)
                loss_dict['specular_pix'] = specular_pix
            return l_total
        else:
            fake_pred = self.net_d(fake_img.detach())    
            real_input = torch.cat([self.inputs,self.svbrdf],dim=1)
            real_pred = self.net_d(real_input)

            l_d = self.cri_gan(real_pred, True, is_disc=True) + self.cri_gan(fake_pred, False, is_disc=True)
            loss_dict['l_d'] = l_d
            loss_dict['real_score'] = real_pred.detach().mean()
            loss_dict['fake_score'] = fake_pred.detach().mean()
            return l_d  

    def HFrequency(self, svbrdf):
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

    def grayImg(self, img):
        '''
            turn RGB images [B, 3, H, W] to Gray images, range(0,255), size[B, 1, H, W]
        '''
        image = torch.unsqueeze(1/3*img[:,0,:,:] + 1/3*img[:,1,:,:] + 1/3*img[:,2,:,:], dim=1)
        return image 

    def decomposition(self, img):
        '''
            turn gray img [B, 3, H, W] to dwt highfrequency[B, 9, H/2, W/2]
        '''
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        f_inputl, f_inputh = self.dwt(img.cpu()) # f_inputh is a list with level elements
        f_inputh[0] = torch.round(f_inputh[0]*100000.0)/100000.0
        HighFrequency = f_inputh[0].cuda() # [B, 3, 3, H/2, W/2]
        HighFrequency = torch.cat([HighFrequency[:,:,0,:,:],HighFrequency[:,:,1,:,:],HighFrequency[:,:,2,:,:]], dim=1) # [B, 3*3, H/2, W/2]
        # HighFrequency = self.norm(HighFrequency)
        return HighFrequency

    def norm(self,x):
        '''
            convert value range to [-1,1]
        '''  
        eps = 1e-20
        return 2.0*((x-torch.min(x))/(torch.max(x)-torch.min(x) + eps)) - 1.0

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.inputs)
            self.output[:,0:3] = torch_norm(self.output[:,0:3],dim=1)
        self.net_g.train()

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

    def save_pre_visuals(self, path, pred):
        output = pred*0.5+0.5
        normal, diffuse, roughness, specular = torch.split(output,[3,3,1,3],dim=1)
        roughness = torch.tile(roughness,[1,3,1,1])
        output = torch.cat([normal,diffuse,roughness,specular],dim=-1)
        output_img = tensor2img(output,rgb2bgr=True)
        imwrite(output_img, path, float2int=False)

    def eval_render(self, pred, gt):
        rerender = self.renderer.render(pred, n_xy=False, keep_dirs=True)
        gtrender = self.renderer.render(gt, n_xy=False, load_dirs=True)
        return rerender, gtrender

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

    def save_rendering(self,path,inputs):
        output_img = tensor2img(inputs*0.5+0.5,rgb2bgr=True)
        imwrite(output_img, path, float2int=False)

    def get_current_visuals(self, pred, gt):
        out_dict = OrderedDict()
        out_dict['predsvbrdf'] = pred.detach().cpu()
        out_dict['gtsvbrdf'] = gt.detach().cpu()
        # out_dict['pred'] = pred_vis
        # out_dict['gt'] = gt_vis
        if hasattr(self, 'brdf'):
            out_dict['brdf'] = self.brdf.detach().cpu()
        return out_dict
        
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name};\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\t'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
    
    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.opt.get('network_d'):
            self.save_network(self.net_d,'net_d',current_iter)
        self.save_training_state(epoch, current_iter)
    