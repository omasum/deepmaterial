
# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
import importlib
import time
import torch
import torch.nn as nn
import math
from deepmaterial.models.sr_model import SRModel
import logging
from deepmaterial.utils import get_root_logger, imwrite, tensor2img
from os import path as osp
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
metric_module = importlib.import_module('deepmaterial.metrics')
from deepmaterial.utils.registry import MODEL_REGISTRY

from deepmaterial.archs.arch_util import input_matrix_wpn

logger = logging.getLogger('basicsr')
@MODEL_REGISTRY.register()
class MetaRDN(SRModel):
    def __init__(self, opt):
        self.opt = opt
        super(MetaRDN, self).__init__(opt)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # dcn_lr_mul = train_opt.get('dcn_lr_mul', 1)
        # logger.info(f'Multiple the learning rate for dcn with {dcn_lr_mul}.')
        # if dcn_lr_mul == 1:
        optim_params = self.net_g.parameters()
        # else:  # separate dcn params and normal params for differnet lr
        #     normal_params = []
        #     dcn_params = []
        #     for name, param in self.net_g.named_parameters():
        #         if 'dcn' in name:
        #             dcn_params.append(param)
        #         else:
        #             normal_params.append(param)
        #     optim_params = [
        #         {  # add normal params first
        #             'params': normal_params,
        #             'lr': train_opt['optim_g']['lr']
        #         },
        #         {
        #             'params': dcn_params,
        #             'lr': train_opt['optim_g']['lr'] * dcn_lr_mul
        #         },
        #     ]

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if current_iter == 1:
            logger.warning('Train all the parameters.')
            for param in self.net_g.parameters():
                param.requires_grad = True
        b,c,h,w = self.lq.shape
        scale = self.net_g.get_scale()
        pos_mat, mask = input_matrix_wpn(h,w,scale)
        self.optimizer_g.zero_grad()
        self.output,_ = self.net_g(self.lq, pos_mat.to(self.device), mask.to(self.device))

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def test(self):
        b,c,h,w = self.lq.shape
        scale = self.net_g.get_scale()
        pos_mat, mask = input_matrix_wpn(h,w,scale)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, _ = self.net_g_ema(self.lq, pos_mat.to(self.device), mask.to(self.device))
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, _ = self.net_g(self.lq, pos_mat.to(self.device), mask.to(self.device))
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
        dataloader.dataset.__getitem__(0)
        for idx, val_data in enumerate(dataloader):
            if 'vimeo' in dataset_name.lower():
                # val_data["lq"] = val_data['lq'][:,3]
                split_result = val_data['lq_path'][0].split('/')
                img_name = (f'{split_result[-4]}_{split_result[-3]}_{split_result[-2]}_'
                                        f'{split_result[-1].split(".")[0]}')
            else:
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            #The output range of meta model is [0,255]. So translate the results to range of [0,1] for convenience
            visuals['result'] = visuals['result']
            if 'gt' in visuals:
                visuals['gt'] = visuals['gt']
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
                # imwrite(gt_img, save_img_path+'_gt.png')

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if 'psnr' in opt_metric.keys():
                    opt_metric['psnr']['scale'] = self.opt['scale'][val_data['scale_idx']]
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(sr_img, gt_img, **opt_)
            if self.opt.get('pbar',True):
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
                break
        if self.opt.get('pbar',True):
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                                tb_logger, scale=self.net_g.get_scale())

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger, scale=None):
        log_str = f'Validation {dataset_name}, scale {scale};\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\t'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def feed_data(self, data):
        if len(data["lq"].shape) == 5:
            data["lq"] = data['lq'][:,3]
        scale_idx = data['scale_idx']
        self.set_scale(scale_idx)
        return super().feed_data(data)

    def set_scale(self, scale_idx):
        self.net_g.set_scale(scale_idx)
