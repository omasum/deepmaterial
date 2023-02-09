import math
import os
import torchvision.utils
import torch

from deepmaterial.data import build_dataloader, build_dataset
from deepmaterial.data.data_sampler import EnlargedSampler


def main(mode='folder'):
    """Test vimeo90k dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
        # gamma_correct: drs
    opt['type'] = 'projDataset'
    opt['phase'] = 'train'
    opt['hd_path'] = 'datasets/embedding.npy'
    opt['y_path'] = 'datasets/predict_prob.npy'
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
        edge_to_x = data['edge_to_x']
        edge_from_x = data['edge_from_x']
        confidence = data['confidence']
        # for j, img in enumerate(render_result**0.4545):
        #     for k, x in enumerate(img):
        #         torchvision.utils.save_image(
        #             x, f'tmp/render_{i:03d}_{k:03d}.png', nrow=nrow, padding=padding, normalize=False)
        #         torchvision.utils.save_image(
        #             data['light'][j][k]/2+0.5, f'tmp/light_{k:03d}.png', nrow=nrow, padding=padding, normalize=False)
        # torchvision.utils.save_image(
        #     data['surface'][j]/2+0.5, f'tmp/surface_{k:03d}.png', nrow=nrow, padding=padding, normalize=False)
        # torchvision.utils.save_image(
        #     data['view'][j]/2+0.5, f'tmp/view_{k:03d}.png', nrow=nrow, padding=padding, normalize=False)


if __name__ == '__main__':
    main()

            