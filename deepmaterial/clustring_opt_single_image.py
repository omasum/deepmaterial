
from logging import root
import time
import cv2
import numpy as np
import torch
from deepmaterial.models import build_model
from deepmaterial.train import init_loggers, parse_options
import os.path as osp
import os
import shutil

from deepmaterial.utils.logger import MessageLogger


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(root_path, is_train=True)
    # initialize loggers
    if opt.get('clear_imgs',False) and osp.exists(opt['path']['visualization']):
        shutil.rmtree(opt['path']['visualization'])
    if opt.get('clear_logs',False) and osp.exists(opt['path']['log']):
        for i in os.listdir(opt['path']['log']):
            t = os.path.join(opt['path']['log'], i)
            if os.path.isfile(t):
                os.remove(t)
    logger, tb_logger = init_loggers(opt)
    msg_logger = MessageLogger(opt, 0, tb_logger)
    
    model = build_model(opt)
    img = (cv2.imread(opt['path']['input_img'])/255)[:,0:256,::-1].astype(np.float32)
    svbrdf = (cv2.imread(opt['path']['input_img'])/255)[:,256:,::-1].astype(np.float32)
    if opt['real_input']:
        model.set_image(img, None)
    else:
        model.set_image(None, svbrdf)
        

    model.initialization()
    total_iter = opt['train']['total_iter']
    pre_loss = 1
    
    logger.info(f'Start optimization!')
    for i in range(total_iter):
        data_time, iter_time = time.time(), time.time()
        model.update_learning_rate(i, warmup_iter=opt['train'].get('warmup_iter', -1))
        model.optimize_parameters(i)
        iter_time = time.time() - iter_time
        if (i+1) % opt['logger']['print_freq'] == 0:
            log_vars = {'epoch': model.nc, 'iter': (i+1)}
            log_vars.update({'lrs': model.get_current_learning_rate()})
            log_vars.update({'time': iter_time, 'data_time': 0})
            log_vars.update(model.get_current_log())
            msg_logger(log_vars)
        if (i+1) % opt['val']['val_freq'] == 0 or i == 0:
            model.save_visual((i+1))
            # break
        # if pre_loss > loss:
        #     pre_loss = loss
        #     model.save_visual((i+1))
        # if (i+1) % opt['val']['opt_freq'] == 0:
        #     model.optimize_material()
        #     model.save_visual((i+1))
        #     break
        # if (i+1) % opt['val']['trace_freq'] == 0:
        #     model.save_trace('tmp/trace.png')
        # if (i+1) % opt['train']['split_freq'] == 0:
        #     error = model.get_error_map()
        #     model.add_prob(error)
        #     logger.info(f'Adding material class at iteration {i+1}, current loss is {loss}!')

if __name__=='__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)