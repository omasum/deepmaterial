import math
import os
import torchvision.utils

from deepmaterial.data import build_dataloader, build_dataset
from deepmaterial.data.data_sampler import EnlargedSampler
from deepmaterial.archs.arch_util import (MeanShift)

from torch.nn import functional as F

def main(mode='folder'):
    """Test vimeo90k dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'Vimeo90K'
    opt['type'] = 'msVimeo90KDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = '/home/hdr/disks/D/Datasets/vimeo_septuplet/sequences'
        opt['dataroot_lq'] = '/home/hdr/disks/D/Datasets/vimeo_septuplet_matlab_X1_X4'  # noqa E501
        opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_Vimeo10K_train_GT.txt'  # noqa E501
        opt['io_backend'] = dict(type='disk')
    elif mode == 'lmdb':
        opt['dataroot_gt'] = ' /home/ext/gaihe/vimeo90kMETA/vimeo90k_train_GT_only4th.lmdb'
        opt['dataroot_lq'] = ' /home/ext/gaihe/vimeo90kMETA/vimeo90k_train_LR7frames.lmdb'
        opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_Vimeo90K_train_GT_00001.txt'  # noqa E501
        opt['io_backend'] = dict(type='lmdb')
    
    opt['num_frame'] = 7
    opt['lq_size'] = 64
    opt['cache_data'] = False
    opt['random_reverse'] = False
    opt['use_flip'] = False
    opt['use_rot'] = False
    opt['use_edge'] = False

    opt['use_shuffle'] = False
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 1
    opt['scale'] = [2.3]

    opt['prefetch_mode'] = 'muti-scale'
    opt['OTFMode'] = 'PIL'
    opt['pin_memory'] = True
    opt['fix_hr'] = False
    opt['dataset_enlarge_ratio'] = 200

    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    train_sampler = EnlargedSampler(dataset, 1,
                                    0, 200)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=train_sampler)

    # a  = data_loader.dataset.__getitem__(0)
    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0
    
    img_range= 1.
    rgb_mean= [0.4488, 0.4371, 0.4040]
    rgb_std= [1.0, 1.0, 1.0]
    print('start...')
    for i, data in enumerate(data_loader):
        if i >= 1:
            break
        # print(data['key'])

        lq = data['lq']
        if opt['use_edge']:
            lq_edge = data['lq_edge']
        gt = data['gt']
        
        # key = data['idx'][0]
        # print(key)
        for j in range(opt['num_frame']):
            # if j != 3:
            #     continue
            
            torchvision.utils.save_image(
                lq[:, j, :, :, :],
                f'tmp/scale{opt["scale"][0]}_lq_{i:03d}_frame{j}.png',
                nrow=nrow,
                padding=padding,
                normalize=False)
            if opt["use_edge"]:
                torchvision.utils.save_image(
                    lq_edge[:, j, :, :],
                    f'tmp/scale{opt["scale"][0]}_lqedge_{i:03d}_frame{j}.png',
                    nrow=nrow,
                    padding=padding,
                    normalize=False)
            bicubic = F.interpolate(
            lq[:, j, :, :, :], scale_factor=opt["scale"][0], mode='bicubic', align_corners=False)
            torchvision.utils.save_image(
                bicubic,
                f'tmp/scale{opt["scale"][0]}_gt_{i:03d}_frame{j}.png',
                nrow=nrow,
                padding=padding,
                normalize=False)


if __name__ == '__main__':
    main()
