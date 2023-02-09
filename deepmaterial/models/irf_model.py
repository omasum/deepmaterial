from collections import OrderedDict
from copy import deepcopy
import logging
from unittest import result
from tqdm import tqdm
from deepmaterial.archs import build_network
from deepmaterial.losses import build_loss
from deepmaterial.utils.img_util import imwrite
from deepmaterial.utils.logger import get_root_logger

from deepmaterial.utils.registry import MODEL_REGISTRY
from deepmaterial.utils.render_util import PlanarSVBRDF, Render, toLDR_torch
from .base_model import BaseModel
import torch
import importlib
import os.path as osp
metric_module = importlib.import_module('deepmaterial.metrics')
logger = logging.getLogger('basicsr')


@MODEL_REGISTRY.register()
class IRFModel(BaseModel):

    def __init__(self, opt):
        super(IRFModel, self).__init__(opt)
        # define network
        self.svbrdf = PlanarSVBRDF(opt['network_g']['brdf_args'])
        self.renderer = Render(opt['network_g'].pop('brdf_args'))
        
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        if not ('print_net_g' in opt and not opt['print_net_g']):
            self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            if 'meta' in opt['name']:
                params = None
            elif self.opt['network_g'].get('with_basicvsrpp', None):
                params = 'state_dict'
            else:
                params = 'params'
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), params)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('cos_opt'):
            self.cri_cos = build_loss(train_opt['cos_opt']).to(self.device)
        else:
            self.cri_cos = None

        if self.cri_pix is None and self.cri_cos is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = self.net_g.parameters()
        # else:  # separate dcn params and normal params for differnet lr
        #     normal_params = []
        #     dcn_params = []
        #     for name, param in self.net_g.named_parameters():
        #         if 'dcn' in name:
        #             dcn_params.append(param)
        #         else:
        #             normal_params.append(param)
            # optim_params = [
            #     {  # add normal params first
            #         'params': normal_params,
            #         'lr': train_opt['optim_g']['lr']
            #     },
            #     {
            #         'params': dcn_params,
            #         'lr': train_opt['optim_g']['lr'] * dcn_lr_mul
            #     },
        #     ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.trace = data['trace']
        self.brdf = data['brdf']
        self.mask = data.get('mask', None)
        # self.nb = data['nb']

    def optimize_parameters(self, current_iter):
        if current_iter == 1:
            logger.warning('Train all the parameters.')
            for param in self.net_g.parameters():
                param.requires_grad = True
    

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.trace, self.mask)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.brdf)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_cos:
            l_cos = self.cri_cos(self.output, self.brdf)
            l_total += l_cos
            loss_dict['l_cos'] = l_cos
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            if self.mask is not None:
                mask = self.mask.to(self.device)
            else:
                mask = None
            self.output = self.net_g(self.trace.to(self.device), mask).cpu()
        self.net_g.train()

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        dataset_name = dataloader.dataset.opt['name']
        batch = dataloader.dataset.opt['len']
        with_metrics = self.opt['val'].get('metrics') is not None
        rb = self.opt['val'].get('save_num', 5)
        normal, mask = self.renderer.sphere_normal(batch=rb)
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
            n = self.output.shape[0]
            rb = min(n, rb)
            if self.opt['val'].get('save_img', False):
                pred = torch.cat([normal[:rb], PlanarSVBRDF.homo2sv(self.output[:rb], normal.shape[-2:])], axis=1)
                gt = torch.cat([normal[:rb], PlanarSVBRDF.homo2sv(self.brdf[:rb], normal.shape[-2:])], axis=1)
                results = self.get_current_visuals(pred, gt)
                img_name = 'visual'
                save_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name)
                if self.opt['is_train']:
                    save_path+=f'_{current_iter}.png'
                else:
                    save_path+='.png'
                self.save_visuals(save_path, results['pred'], results['gt'])
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                # print('pred:',self.output)
                # print('gt:',self.brdf)
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    if metric_type == 'cos':
                        error = self.cri_cos(self.output, self.brdf)*opt_.pop('weight')
                    elif metric_type == 'pix':
                        error = torch.abs(self.output-self.brdf).mean()*opt_.pop('weight')
                    self.metric_results[name] += error
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

    def save_visuals(self, path, pred, gt, toLDR=True):
        n,b,c,h,w = pred.shape
        if toLDR:
            pred = toLDR_torch(pred, sturated=True)
            gt = toLDR_torch(gt, sturated=True)
        pred = pred.permute(3,0,1,4,2).contiguous().view(h,-1,c)
        gt = gt.permute(3,0,1,4,2).contiguous().view(h,-1,c)
        imgs = torch.cat([pred, gt], dim=0).numpy()
        imwrite(imgs[:,:,::-1], path, float2int=False)

    def get_current_visuals(self, pred, gt):
        out_dict = OrderedDict()
        pred_vis = self.renderer.render(pred, r_single=False, random_light=False)
        gt_vis = self.renderer.render(gt, r_single=False, random_light=False)
        out_dict['trace'] = self.trace.detach().cpu()
        out_dict['output'] = self.output.detach().cpu()
        out_dict['pred'] = pred_vis
        out_dict['gt'] = gt_vis
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
        self.save_training_state(epoch, current_iter)