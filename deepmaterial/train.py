import argparse
import datetime
import logging
import math
import random
import time
from os import path as osp
import os, traceback, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import setproctitle

from deepmaterial.utils.misc import cp_options

proc_title = "ZLHTrain"
setproctitle.setproctitle(proc_title)
torch.backends.cudnn.benchmark = False
# print(f"there is {torch.cuda.device_count()} available gpu device")
from deepmaterial.data import build_dataloader, build_dataset
from deepmaterial.data.data_sampler import EnlargedSampler
from deepmaterial.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from deepmaterial.models import build_model 
from deepmaterial.utils import (MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename, set_random_seed)
from deepmaterial.utils.dist_util import get_dist_info, init_dist
from deepmaterial.utils.options import dict2str, parse

def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=is_train)
    
    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        # print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt, args.opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='deepmaterial', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: ' f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def train_pipeline(root_path):
    start_time = time.time()
    # parse options, set distributed setting, set ramdom seed
    opt, opt_path = parse_options(root_path, is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # copy the option file to experiment folder and rename old option file
    cp_options(opt, opt_path)

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result
    # train_loader.dataset.set_scale(15)
    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = build_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = build_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda' or prefetch_mode == 'muti-scale':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                        "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            train_sampler.set_epoch(epoch)
            prefetcher.reset()
            train_data = prefetcher.next()
            
            while train_data is not None:
                start = time.time()
                data_time = time.time() - data_time

                current_iter += 1
                if current_iter > total_iters:
                    break
                # update learning rate
                model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                # training
                train_data['iter'] = current_iter
                model.feed_data(train_data)
                # print(current_iter, torch.cuda.memory_reserved()+torch.cuda.memory_allocated(), train_data['trace'].shape[1], torch.cuda.max_memory_allocated())
                model.optimize_parameters(current_iter)
                iter_time = time.time() - iter_time
                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_time, 'data_time': data_time})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

                # save models and training states
                if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    model.save(epoch, current_iter)

                # validation
                if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

                data_time = time.time()
                iter_time = time.time()
                train_data = prefetcher.next()
                # print("iteration time:",time.time()-start)
                # print("----------------iteration-------------------------")
            # end of iter

        # end of epoch
        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        

        logger.info(f'End of training. Time consumed: {consumed_time}')
        if 'debug' in opt['name']:
            return 
        logger.info('Save the latest model.')
        model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
        
        if opt.get('log_scale', False):
            scale_log = sorted(model.net_g.scale_log)
            scale_str = ""
            for key in scale_log:
                value = model.net_g.scale_log[key]
                scale_str += str(model.net_g.upsample_meta.scale[key]) + ":" + str(value) + ", "
            logger.info(scale_str)

        if opt.get('val') is not None and not 'test' in opt.get('name'):
            model.validation(val_loader, total_iters, tb_logger, opt['val']['save_img'])
        if tb_logger:
            tb_logger.close()
    except KeyboardInterrupt as e:
        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f'Exception at iteration: {current_iter}, epoch:{epoch}. Time consumed: {consumed_time}')
        logger.info('Save the model.')
        model.save(epoch, current_iter)
    # except Exception as e:
    #     logger.info('repr(e):\t ' + repr(e))
    #     logger.info('traceback.format_exc():\n%s' % traceback.format_exc())
    #     logger.info('Save the model.')
    #     model.save(epoch, current_iter)

        

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    root_path = "/home/cjm/DeepMaterial"
    train_pipeline(root_path)


# conda activate ghpy37
# cd /mnt/hard_disk/gaihe/code/BasicSR-master
# python deepmaterial/train.py -opt options/train/VW/train_vw.yml