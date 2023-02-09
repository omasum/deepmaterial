import math
import os
import torchvision.utils
import torch

from deepmaterial.data import build_dataloader, build_dataset
from deepmaterial.data.data_sampler import EnlargedSampler
from deepmaterial.utils.render_util import PlanarSVBRDF


def main(mode='folder'):
    """Test vimeo90k dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    brdf_args = {}
    brdf_args['split_num'] = 5
    brdf_args['nbRendering'] = 1
    brdf_args['split_axis'] = 1
    brdf_args['concat'] = True
    brdf_args['svbrdf_norm'] = True
    brdf_args['size'] = 256
    brdf_args['order'] = 'pndrs'
    brdf_args['permute_channel'] = True
    brdf_args['toLDR'] = False
    brdf_args['useAug']=True
        # gamma_correct: drs
    opt['brdf_args'] = brdf_args
    opt['phase'] = 'train'
    opt['name'] = 'brdf'
    opt['type'] = 'homoDataset'
    opt['len'] = 100
    opt['range_d'] = [0,1]
    opt['range_s'] = [0,1]
    opt['range_r'] = [0,1]
    opt['range_n'] = {"theta":[0,70],"phi":[0,360]}
    opt['nb'] = 3000
    opt['rand_nb'] = False
    opt['rand_lv'] = True
    opt['mix_plane'] = True
    opt['plane_rate'] = 0.3

    # data loader
    opt['use_shuffle'] = False
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 4
    opt['dataset_enlarge_ratio'] = 1
    opt['prefetch_mode'] = 'cuda'
   
    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    train_sampler = EnlargedSampler(dataset, 1,
                                    0, 1)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, sampler=train_sampler)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    # data = dataset.__getitem__(0)
    for i, data in enumerate(data_loader):
        if i > 10:
            break
        print(data['trace'].shape)
        render_result = data['inputs']
        # render_result, _, _, _, _ = dataset.svBRDF_utils.render(svbrdfs, light_dir=data['light'], view_dir=data['view'], light_dis=data['dis'], n_xy=True)
        svbrdfs = PlanarSVBRDF.brdf2uint8(data['svbrdf'], n_xy=False)
        torchvision.utils.save_image(
            svbrdfs, f'tmp/brdf_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(
            (render_result**0.4545), f'tmp/render_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        # torchvision.utils.save_image(
        #     data['spmask']*1.0, f'tmp/mask_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)


if __name__ == '__main__':
    main()

            