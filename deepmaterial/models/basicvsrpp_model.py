from numpy.core.fromnumeric import size
from deepmaterial.utils.img_util import img2edge
import torch
import logging
from torch.nn.parallel import DistributedDataParallel
from deepmaterial.models.video_base_model import VideoBaseModel
import importlib
import time
import torch.nn as nn
import math
import logging
from deepmaterial.utils import get_root_logger, imwrite, tensor2img, imresize_PIL, feat2img_fast
import numpy as np
from os import path as osp
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
from collections import Counter
from deepmaterial.utils.dist_util import get_dist_info
from torch.nn import functional as F
logger = logging.getLogger('basicsr')
metric_module = importlib.import_module('deepmaterial.metrics')
from deepmaterial.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class BasicVsrPPModel(VideoBaseModel):
    def __init__(self, opt):
        super().__init__(opt)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = self.net_g.parameters()
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int): Warmup iter numbers. -1 for no warmup.
                Default: -1.
        """
        if current_iter > 1 and current_iter % self.opt['accumulation_steps'] == 0:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def optimize_parameters(self, current_iter):
        mid = time.time()
        if current_iter == 1:
            logger.warning('Train all the parameters.')
            for param in self.net_g.parameters():
                param.requires_grad = True
            if isinstance(self.net_g, DistributedDataParallel):
                logger.warning('Set net_g.find_unused_parameters = False.')
                self.net_g.find_unused_parameters = False

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # if self.cri_edge: 
        #     re_lr = F.interpolate(
        #         self.output, size=(self.lq.shape[2], self.lq.shape[3]), mode='bilinear', align_corners=False)
        #     re_edge = img2edge(re_lr)
        #     l_edge = self.cri_edge(re_edge, self.lq_edge)
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # 梯度累加，变相提升batch_size，相当于accumulate_steps完成一个batch的训练
        accumulation_steps = self.opt['accumulation_steps']
        l_total = l_total/accumulation_steps

        l_total.backward()
        if((current_iter+1)%accumulation_steps)==0:
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()

        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            if self.net_g.with_edge:
                self.output = self.net_g([self.lq, self.lq_edge])
            elif self.net_g.is_bicubic:
                scale = self.opt['scale'][self.scale_idx]
                self.output = F.interpolate(
                self.lq[:,self.net_g.center_frame_idx,:,:,:], scale_factor=scale, mode='bicubic', align_corners=False)
            else:
                self.output = self.net_g(self.lq)
            if 'fixed_net_g_scale' in self.opt and not self.net_g.is_bicubic:
                net_g_scale = self.opt['fixed_net_g_scale']
                b,c,h,w = self.gt.shape
                scale = self.opt['scale'][self.scale_idx]
                self.output = F.interpolate(
                self.output, scale_factor=scale/net_g_scale, mode='bicubic', align_corners=False)

                N, _, H, W = self.output.size()
                self.gt = self.gt[:,:,:H,:W]
        self.net_g.train()

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics and not hasattr(self, 'metric_results'):
            self.metric_results = {}
            num_frame_each_folder = Counter(dataset.data_info['folder'])
            for folder, num_frame in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(
                    num_frame,
                    len(self.opt['val']['metrics']),
                    dtype=torch.float32,
                    device='cuda')
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()
        # record all frames (border and center frames)
        if rank == 0 and self.opt['pbar']:
            pbar = tqdm(total=len(dataset), unit='frame')
        for idx in range(rank, len(dataset), world_size):
            val_data = dataset[idx]
            if 'ms' in dataset.opt["name"].lower():
                val_data['scale_idx'] = dataset.scale_idx
            else:
                val_data['scale_idx'] = -1
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            folder = val_data['folder']
            frame_idx, max_idx = val_data['idx'].split('/')
            lq_path = val_data['lq_path']

            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            if self.net_g.vis_feature:
                feat2img_fast(visuals["result"], 'all', save_path="./tmp")
                continue
            result_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    raise NotImplementedError(
                        'saving image is not supported during training.')
                else:
                    if 'vimeo' in dataset_name.lower():  # vimeo90k dataset
                        split_result = lq_path.split('/')
                        if 'ms' in dataset_name.lower():
                            img_name = (f'{split_result[-4]}_{split_result[-3]}_{split_result[-2]}_'
                                        f'{split_result[-1].split(".")[0]}')
                        else:
                            img_name = (f'{split_result[-3]}_{split_result[-2]}_'
                                        f'{split_result[-1].split(".")[0]}')
                    else:  # other datasets, e.g., REDS, Vid4
                        img_name = osp.splitext(osp.basename(lq_path))[0]

                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            folder,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            folder, f'{img_name}_{self.opt["name"]}.png')
                imwrite(result_img, save_img_path)
                # imwrite(gt_img, save_img_path+'_gt.png')

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if 'psnr' in opt_metric.keys() and (self.net_g.with_meta or self.net_g.with_resMeta or self.net_g.with_onlyMeta):
                    opt_metric['psnr']['scale'] = self.opt['scale'][val_data['scale_idx']]
                for metric_idx, opt_ in enumerate(opt_metric.values()):
                    metric_type = opt_.pop('type')
                    result = getattr(metric_module,
                                     metric_type)(result_img, gt_img, **opt_)
                    self.metric_results[folder][int(frame_idx),
                                                metric_idx] += result

            # progress bar
            if rank == 0 and self.opt['pbar']:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(
                        f'Test {folder}:'
                        f'{int(frame_idx) + world_size}/{max_idx}')
        if rank == 0 and self.opt['pbar']:
            pbar.close()

        if with_metrics:
            # if self.opt['dist']:
            #     # collect data among GPUs
            #     for _, tensor in self.metric_results.items():
            #         dist.reduce(tensor, 0)
            #     dist.barrier()
            # else:
                # pass  # assume use one gpu in non-dist testing

            if rank == 0:
                scale = dataloader.dataset.opt["scale"][-1] if not ("otf" in dataloader.dataset.opt["type"].lower() \
                    or "ms" in dataloader.dataset.opt["type"].lower()) else dataloader.dataset.scale[dataloader.dataset.scale_idx]
                self._log_validation_metric_values(current_iter, dataset_name,
                                                   tb_logger, scale=scale)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger, scale=4.0):
        # average all frames for each sub-folder
        # metric_results_avg is a dict:{
        #    'folder1': tensor (len(metrics)),
        #    'folder2': tensor (len(metrics))
        # }
        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }
        # total_avg_results is a dict: {
        #    'metric1': float,
        #    'metric2': float
        # }
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()
        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)
        log_str = f'Validation {dataset_name}, scale {scale};\t'
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
            log_str += '\t'
        log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                for folder, tensor in metric_results_avg.items():
                    tb_logger.add_scalar(f'metrics/{metric}/{folder}', tensor[metric_idx].item(), current_iter)
    def feed_data(self, data):
        if self.net_g.with_meta or self.net_g.with_uwlb or 'fixed_net_g_scale' in self.opt or self.net_g.with_resMeta or self.net_g.with_resNfMeta or self.net_g.with_onlyMeta:
            self.scale_idx = data['scale_idx']
            self.set_scale(self.scale_idx)
        if 'lq_edge' in data:
            self.lq_edge = data['lq_edge'].to(self.device)
        if not self.scale_idx in self.net_g.scale_log.keys():
            self.net_g.scale_log[self.scale_idx] = 0
        else:
            self.net_g.scale_log[self.scale_idx] = self.net_g.scale_log[self.scale_idx]+1
        return super().feed_data(data)

    def set_scale(self, scale_idx):
        if self.net_g.with_meta:
            self.net_g.upsample_meta.set_scale(scale_idx)
        elif self.net_g.with_uwlb:
            self.net_g.UWLB.set_scale(scale_idx)
        if self.net_g.with_scaleAware:
            for i in range(len(self.net_g.reconstruction)):
                self.net_g.reconstruction[i].set_scale(scale_idx)
        if self.net_g.with_resMeta or self.net_g.with_resNfMeta or self.net_g.with_onlyMeta:
            self.net_g.upsample_meta_res.set_scale(scale_idx)