import math
import os
import torchvision.utils

from deepmaterial.data import build_dataloader, build_dataset


def main(mode='folder'):
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'DIV2K'
    opt['type'] = 'PairedImageDatasetMETA'
    if mode == 'folder':
        opt['dataroot_gt'] = 'C:/Datasets/DIV2K/DIV2K_train_HR'
        opt['dataroot_lq'] = 'D:/Datasets/DIV2K/DIV2K_matlabX1_X4'
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')

    opt['lq_size'] = 256
    opt['use_flip'] = False
    opt['use_rot'] = False

    opt['use_shuffle'] = False
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 1
    opt['scale'] = [2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]

    opt['dataset_enlarge_ratio'] = 1

    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 0:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        lq_path = data['lq_path']
        gt_path = data['gt_path']
        print(lq_path, gt_path)
        torchvision.utils.save_image(lq, f'tmp/lq_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(gt, f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)


if __name__ == '__main__':
    main()
