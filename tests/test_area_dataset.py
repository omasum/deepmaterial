import math
import os
from tkinter.tix import Tree
import torchvision.utils

from deepmaterial.data import build_dataloader, build_dataset
from deepmaterial.data.data_sampler import EnlargedSampler


def main(mode='folder'):
    """Test vimeo90k dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    brdfArgs = {}
    lightingConfig = {}
    brdfArgs['nbRendering'] = 1
    brdfArgs['size'] = 256
    brdfArgs['order'] = 'pndrs'
    brdfArgs['toLDR'] = True
    brdfArgs['lampIntensity'] = 3
    lightingConfig['texture'] = 'tmp/0.png'
    lightingConfig['textureMode'] = 'Torch'
    lightingConfig['type'] = 'RectLighting'

    # gamma_correct: drs
    opt['brdf_args'] = brdfArgs
    opt['light_args'] = lightingConfig
    opt['phase'] = 'train'
    opt['name'] = 'brdf'
    opt['type'] = 'areaDataset'
    opt['data_path'] = '/home/sda/svBRDFs/trainBlended'
    # opt['data_path'] = '/home/xh/cjm/18single_image/Data_Deschaintre18/testBlended' #2080ti
    opt['rendering_path'] = '/home/sda/svBRDFs/'
    opt['input_mode'] = 'render'
    opt['light_mode'] = 'area'
    opt['log'] = False

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
    dataset.__getitem__(0)
    over_exposed = 0
    for i, data in enumerate(data_loader):
        if i > 20:
            break
        print(data['name'])
        c, h, w = data['imgs'].shape[-3:]
        render_result = data['imgs'].view(-1, c, h, w)
        torchvision.utils.save_image(
            render_result, f'tmp/render_{i:03d}.png', nrow=render_result.shape[0], padding=padding, normalize=False)
        svbrdfs = data['svbrdfs']
        svbrdfs = dataset.svbrdf.brdf2uint8(svbrdfs, n_xy=False, r_single=True, gamma_correct='ds')
        torchvision.utils.save_image(
            svbrdfs, f'tmp/brdf_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        if (i+1)% 10000 == 0:
            print(f'{i+1} images have been scanned!')
    print(over_exposed/len(dataset))

if __name__ == '__main__':
    main()

            