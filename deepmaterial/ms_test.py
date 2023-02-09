
import logging
import torch
from os import path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from deepmaterial.data import build_dataloader, build_dataset
from deepmaterial.models import build_model
from deepmaterial.train import parse_options
from deepmaterial.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from deepmaterial.utils.options import dict2str


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='deepmaterial', log_level=logging.INFO, log_file=log_file)
    # logger.info(get_env_info())
    # logger.info(dict2str(opt))
    opt['scale'] = opt['scale']
    scale = opt['scale']
    for s in scale:
        # create test dataset and dataloader
        test_loaders = []
        for phase, dataset_opt in sorted(opt['datasets'].items()):
            dataset_opt["fixed_scale"] = s
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            # logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
            test_loaders.append(test_loader)

        # create model
        model = build_model(opt)

        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            # logger.info(f'Testing {test_set_name}...')
            model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        torch.cuda.empty_cache()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)

# python deepmaterial/ms_test.py -opt options/test/MSVR/test_MSVR_Pcd4FuseMeta_L1_F7G064Feat64.yml