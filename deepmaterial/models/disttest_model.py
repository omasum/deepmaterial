import torch
import logging
from torch.nn.parallel import DistributedDataParallel
from deepmaterial.models.video_base_model import VideoBaseModel
import importlib
import time
import torch
import logging
from collections import OrderedDict
logger = logging.getLogger('basicsr')
metric_module = importlib.import_module('deepmaterial.metrics')
from deepmaterial.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class DistTestModel(VideoBaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.net_arg = self.opt["network_g"]

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = self.net_g.parameters()

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
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

        inputs = self.lq
        self.output = self.net_g(inputs)
        # print("forward time:",time.time()-start)
        
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        l_total.backward() # 2267
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()

        self.log_dict = self.reduce_loss_dict(loss_dict)
    