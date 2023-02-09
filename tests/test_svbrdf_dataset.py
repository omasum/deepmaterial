import math
import os
import torchvision.utils

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
    brdf_args['nbRendering'] = 2
    brdf_args['split_axis'] = 1
    brdf_args['concat'] = True
    brdf_args['svbrdf_norm'] = True
    brdf_args['size'] = 256
    brdf_args['order'] = 'pndrs'
    brdf_args['permute_channel'] = True
    brdf_args['toLDR'] = False
    # brdf_args['lampIntensity'] = 0.3

        # gamma_correct: drs
    opt['brdf_args'] = brdf_args
    opt['phase'] = 'test'
    opt['name'] = 'brdf'
    opt['type'] = 'svbrdfDataset'
    # opt['dataroot_train'] = 'D:/Datasets/svBRDFs/testBlended'
    # opt['dataroot_test'] = 'D:/Datasets/svBRDFs/testBlended'
    opt['data_path'] = '/home/sda/svBRDFs/testBlended'
    # opt['data_path'] = '/home/xh/cjm/18single_image/Data_Deschaintre18/testBlended' # 2080ti
    opt['fixed_input'] = False
    # opt['fixed_path'] = 'D:/Datasets/svBRDFs/testBlended/0000033;brick_uneven_stonesXPolishedMarbleFloor_01;2Xdefault.png'
    # opt['fixed_path'] = 'D:/VSProject/BasicSR-master/experiments/001_CSME/visualization/render_TestImg_200.png'
    # opt['fixed_path'] = '/home/sda/svBRDFs/testBlended/0000033;brick_uneven_stonesXPolishedMarbleFloor_01;2Xdefault.png'
    opt['io_backend'] ={}
    opt['io_backend']['type'] = 'disk'
    opt['input_type'] = 'brdf'

    opt['use_flip'] = False
    opt['use_rot'] = False
    opt['use_ambient'] = False
    opt['jitterlight'] = False

    # data loader
    opt['use_shuffle'] = False
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 1
    opt['dataset_enlarge_ratio'] = 1
    opt['prefetch_mode'] = None
   
    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    train_sampler = EnlargedSampler(dataset, 1,
                                    0, 1)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, sampler=train_sampler)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 4:
            break
        print(data['path'])
        render_result = data['inputs']
        svbrdfs = data['svbrdfs']
        # render_result, _, _, _, _ = dataset.svBRDF_utils.render(svbrdfs, light_dir=data['light'], view_dir=data['view'], light_dis=data['dis'], n_xy=True)
        svbrdfs = PlanarSVBRDF.brdf2uint8(svbrdfs, n_xy=True, gamma_correct='ds')
        torchvision.utils.save_image(
            svbrdfs, f'tmp/brdf_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        for j, img in enumerate(render_result**0.4545):
            for k, x in enumerate(img):
                torchvision.utils.save_image(
                    x, f'tmp/render_{i:03d}_{k:03d}.png', nrow=nrow, padding=padding, normalize=False)
                torchvision.utils.save_image(
                    data['light'][j][k]/2+0.5, f'tmp/light_{k:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(
            data['surface'][j]/2+0.5, f'tmp/surface_{k:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(
            data['view'][j]/2+0.5, f'tmp/view_{k:03d}.png', nrow=nrow, padding=padding, normalize=False)


if __name__ == '__main__':
    main()

            