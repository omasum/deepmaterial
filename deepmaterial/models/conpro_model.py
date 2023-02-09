from collections import OrderedDict
from copy import deepcopy
import logging
from turtle import position

import tqdm

from deepmaterial.archs import build_network
from deepmaterial.archs.arch_util import arg_softmax, gumbel_softmax
from deepmaterial.losses import build_loss
from deepmaterial.utils.img_util import color_map, imwrite
import torch.nn.functional as F
from deepmaterial.utils.logger import get_root_logger

from deepmaterial.utils.registry import MODEL_REGISTRY
from deepmaterial.models.base_model import BaseModel
import torch
import importlib
import os.path as osp
metric_module = importlib.import_module('deepmaterial.metrics')
logger = logging.getLogger('deepmaterial')


@MODEL_REGISTRY.register()
class ConProModel(BaseModel):

    def __init__(self, opt):
        super(ConProModel, self).__init__(opt)
        # define network
        self.net_pro = build_network(opt['network_pro'])
        self.net_con = build_network(opt['network_con'])

        self.net_pro = self.model_to_device(self.net_pro)
        self.net_con = self.model_to_device(self.net_con)
        if not ('print_net_g' in opt and not opt['print_net_g']):
            self.print_network(self.net_pro)
            self.print_network(self.net_con)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_pro', None)
        if load_path is not None:
            self.load_network(self.net_pro, load_path, self.opt['path'].get('strict_load_pro', True))
            logger.info('Load pre-trained parameters from '+load_path)
        load_path = self.opt['path'].get('pretrain_network_con', None)
        if load_path is not None:
            self.load_network(self.net_con, load_path, self.opt['path'].get('strict_load_con', True))
            logger.info('Load pre-trained parameters from '+load_path)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_pro.train()
        self.net_con.train()
        train_opt = self.opt['train']

        # define losses
        self.cri_con = build_loss(train_opt['pixel_opt']).to(self.device)

        if self.cri_con is None and self.cri_cos is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = self.net_pro.parameters()
        optim_type = train_opt['optim_pro'].pop('type')
        self.optimizer_pro = self.get_optimizer(optim_type, optim_params, **train_opt['optim_pro'])
        
        self.optimizers.append(self.optimizer_pro)

        optim_params = self.net_con.parameters()
        optim_type = train_opt['optim_con'].pop('type')
        self.optimizer_con = self.get_optimizer(optim_type, optim_params, **train_opt['optim_con'])

        self.optimizers.append(self.optimizer_con)

    def feed_data(self, data):
        self.hd = data['hd']
        self.ld = data.get('ld', None)
        self.confidence = data.get('confidence', None)

    def optimize_parameters(self, current_iter):
        if current_iter == 1:
            logger.warning('Train all the parameters.')
            for param in self.net_pro.parameters():
                param.requires_grad = True
            for param in self.net_con.parameters():
                param.requires_grad = True
    
        self.optimizer_pro.zero_grad()
        self.optimizer_con.zero_grad()

        self.pos = self.net_pro(self.hd)
        self.y = self.net_con(self.pos)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_con:
            l_con = self.cri_con(self.y, self.confidence)
            l_total += l_con
            loss_dict['l_con'] = l_con
        l_total.backward()
        self.optimizer_pro.step()
        self.optimizer_con.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def projection(self):
        self.net_pro.eval()
        with torch.no_grad():
            position = self.net_pro(self.hd).cpu()
        self.net_pro.train()
        return position

    def division(self, position):
        self.net_con.eval()
        with torch.no_grad():
            confidence = self.net_con(position).cpu()
        self.net_con.train()
        return confidence

    def test(self):
        position = self.projection()
        confidence = self.division(position=position)
        return position, confidence

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
        pos = []
        confi = []
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            position, confidence = self.test()
            torch.cuda.empty_cache()
            pos.append(position)
            confi.append(confidence)

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
        
        if self.opt['val'].get('save_img', False):
            results = self.get_current_visuals(pos, confi)
            img_name = 'visual'
            save_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name)
            if self.opt['is_train']:
                save_path+=f'_{current_iter}.png'
            else:
                save_path+='.png'
            self.save_visuals(save_path, results['pred'], results['gt'])

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                                tb_logger)

    def save_visuals(self, path, pred, gt, toLDR=True):
        n,b,c,h,w = pred.shape
        pred = pred.permute(3,0,1,4,2).contiguous().view(h,-1,c)
        gt = gt.permute(3,0,1,4,2).contiguous().view(h,-1,c)
        imgs = torch.cat([pred, gt], dim=0).numpy()
        imwrite(imgs[:,:,::-1], path, float2int=False)

    def get_current_visuals(self, position, confidence):
        pass
        
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
        self.save_network(self.net_pro, 'net_pro', current_iter)
        self.save_network(self.net_con, 'net_con', current_iter)
        self.save_training_state(epoch, current_iter)