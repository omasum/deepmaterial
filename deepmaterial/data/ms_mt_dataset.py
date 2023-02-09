import random, copy
from cv2 import resize
import torch, math
from pathlib import Path
from torch.utils import data as data

from os import path as osp
import glob

from deepmaterial.data.transforms import augment, paired_random_crop, paired_random_crop_mutiscale
from deepmaterial.utils import FileClient, get_root_logger, imfrombytes, img2tensor, scandir, imresize, imwrite, tensor2img, img2edge,imresize_PIL
from deepmaterial.data.video_test_dataset import VideoTestVimeo90KDataset,VideoTestDataset
from deepmaterial.data.vimeo90k_dataset import Vimeo90KDataset
from deepmaterial.data.paired_image_dataset import PairedImageDataset
from torchvision.transforms.functional import normalize
from torch.nn import functional as F
from deepmaterial.data.data_util import (read_img_seq, generate_frame_indices, muti_scale_paired_paths_from_folder)
from deepmaterial.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class msmtVimeo90KDataset(Vimeo90KDataset):
    def __init__(self, opt):
        self.scale = opt['scale']
        self.fix_hr = opt['fix_hr']
        self.scale_idx = 0
        super(msmtVimeo90KDataset,self).__init__(opt)
        self.use_edge = True if 'use_edge' in self.opt and self.opt['use_edge'] else False
        # if opt['useDir']:
        #     for i, lq in enumerate(self.data_info['lq_path']):
        #         root, subpath = lq[self.opt['num_frame'] // 2].split('x'+str(scale))
        #         self.data_info['lq_path'][i] = [root + 'x' + str(scale) + '/sequences' + subpath]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
            self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.get_scale()
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the GT frame (im4.png), generated from lqs
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{key}/im4'
            else:
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=False)
            img_gts.append(img_gt)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path_gt = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path_gt = self.lq_root / clip / seq / str(scale) / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path_gt, 'lq')
            img_lq_gt = imfrombytes(img_bytes, float32=False)
            img_lqs.append(img_lq_gt)
        #如果使用边缘信息当做输入，则数据集生成边缘信息
        if self.use_edge:
            img_lqs_edge = img2edge(img_lqs, keep_dims = True)
        else:
            img_lqs_edge = None

        img_gt, img_lqs, img_lqs_edge = self.get_patch(img_lqs, img_gts, img_lqs_edge)
        
        img_lqs = img2tensor(img_lqs, normalization=True)
        img_gt = img2tensor(img_gt, normalization=True)

        img_lqs = torch.stack(img_lqs, dim=0)
        img_gt = torch.stack(img_gt, dim=0)
        results = {'lq': img_lqs, 'gt': img_gt, 'key': key}
        if self.use_edge:
            
            img_lqs_edge = img2tensor(img_lqs_edge, normalization=True, singleChannel = True)
            img_lqs_edge = torch.stack(img_lqs_edge, dim=0)
            results ['lq_edge'] = img_lqs_edge
        if 'float16' in self.opt and self.opt['float16']:
            for k, v in zip(results.keys(), results.values()):
                if torch.is_tensor(v):
                    results[k] = v.half()
        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return results

    def get_patch(self, img_lq, img_gt, img_lq_edge=None):
        scale = self.scale[self.scale_idx]
        # multi_scale = len(self.scale) > 1
        if self.opt['phase']=='train':
            lq_patch_size = self.opt['lq_size']
            lq_length = len(img_lq)
            results = paired_random_crop_mutiscale(
                img_lq,
                img_gts=img_gt,
                img_lqs_edge=img_lq_edge,
                patch_size=lq_patch_size,
                scale=scale
            )
            
            if img_lq_edge is not None:
                if lq_length != 1:
                    img_gt, img_lq, img_lq_edge = results
                    img_lq.extend(img_lq_edge)
                    img_lq.extend(img_gt)
                else:
                    img_gt, img_lq = [img_lq, img_lq_edge]
                    img_lq.extend(img_gt)
            else:
                img_gt, img_lq = results
                if lq_length == 1:
                    img_lq = [img_lq]
                img_lq.extend(img_gt)
                
            img_lq = augment(img_lq, hflip = self.opt['use_flip'],
                                    rotation = self.opt['use_rot'])
            
            if img_lq_edge is not None:
                img_lq_edge = img_lq[lq_length:2*lq_length]
                img_gt = img_lq[2*lq_length]
            else:
                img_gt = img_lq[lq_length:2*lq_length]
            img_lq = img_lq[0:lq_length]

        results = [img_gt, img_lq, img_lq_edge]

        return results

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
    def get_scale(self):
        return self.scale[self.scale_idx]

@DATASET_REGISTRY.register()
class msmtVimeo90KDatasetTest(data.Dataset):
    def __init__(self, opt):
        self.scale = opt['scale']
        self.bicubic_up = opt.get('bicubic_up',False)
        if 'fixed_scale' in opt:
            self.scale_idx = self.scale.index(opt['fixed_scale'])
        else:
            self.scale_idx = 0
        super(msmtVimeo90KDatasetTest, self).__init__()
        self.opt = opt
        self.use_edge = True if 'use_edge' in self.opt and self.opt['use_edge'] else False
        self.cache_data = opt['cache_data']
        if self.cache_data:
            raise NotImplementedError(
                'cache_data in Vimeo90K-Test dataset is not implemented.')
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'folder': [],
            'idx': [],
            'border': []
        }
        neighbor_list = [
            i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])
        ]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt[
            'type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        # logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = [
                osp.join(self.gt_root, subfolder, f'im{i}.png')
                for i in neighbor_list
            ]
            self.data_info['gt_path'].append(gt_path)
            lq_path = [
                osp.join(self.lq_root, subfolder, f'im{i}.png')
                for i in neighbor_list
            ]
            self.data_info['lq_path'].append(lq_path)

            self.data_info['folder'].append('vimeo90k')
            self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
            self.data_info['border'].append(0)

    def __getitem__(self, index):
        scale = self.get_scale()
        lq_path = []
        for path in self.data_info['lq_path'][index]:
            clip, seq, img = path.split('/')[-3:]
            img_lq_path = self.lq_root+ '/' +clip+ '/' +seq+ '/' +str(scale)+ '/' +img
            lq_path.append(img_lq_path)
        gt_path = self.data_info['gt_path'][index]
        imgs_lqs = read_img_seq(lq_path, numpy=True)
        img_gt = read_img_seq(gt_path, numpy=True)
        lr_h, lr_w,_ = imgs_lqs[0].shape
        hr_h = int(lr_h*scale)
        hr_w = int(lr_w*scale)
        img_gt = [img[:hr_h, :hr_w,:] for img in img_gt]
        if self.bicubic_up:
            # imgs_lqs = []
            pass

        if self.use_edge:
            img_lqs_edge = img2edge(imgs_lqs, keep_dims = True)
            
        img_gt = img2tensor(img_gt, normalization=True)
        img_gt = torch.stack(img_gt,dim=0)
        imgs_lqs = torch.stack(img2tensor(imgs_lqs, normalization=True),dim=0)
        results = {
            'lq': imgs_lqs,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': self.data_info['folder'][index],  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/843
            'border': self.data_info['border'][index],  # 0 for non-border
            'lq_path': lq_path[self.opt['num_frame'] // 2]  # center frame
        }
        if self.use_edge:
            img_lqs_edge = img2tensor(img_lqs_edge, normalization=True, singleChannel = True)
            img_lqs_edge = torch.stack(img_lqs_edge, dim=0)
            results ['lq_edge'] = img_lqs_edge
        if 'fixed_scale' in self.opt:
            results['scale_idx'] = self.scale_idx
        return results

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
    def get_scale(self):
        return self.scale[self.scale_idx]
    def __len__(self):
        return len(self.data_info['gt_path'])

@DATASET_REGISTRY.register()
class msmtVid4(data.Dataset):
    def __init__(self, opt):
        super(msmtVid4, self).__init__()
        self.opt = opt
        self.scale=opt['scale']
        self.scale_idx = self.scale.index(opt['fixed_scale'])
        self.bicubic_up = opt.get('bicubic_up',False)
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        if opt['name'].lower() in ['msvid4', 'reds4', 'redsofficial']:
            for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_lq)
                img_paths_lq = sorted(list(scandir(subfolder_lq+'/X%.2f'%opt['fixed_scale'], full_path=True)))
                img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

                max_idx = len(img_paths_lq)
                assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                      f' and gt folders ({len(img_paths_gt)})')

                self.data_info['lq_path'].extend(img_paths_lq)
                self.data_info['gt_path'].extend(img_paths_gt)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')
                border_l = [0] * max_idx
                for i in range(self.opt['num_frame'] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                    self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt
        else:
            raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][i] for i in select_idx])
            # img_gt.squeeze_(0)
        _, lr_h, lr_w = imgs_lq[0].shape
        hr_h = int(lr_h*self.scale[self.scale_idx])
        hr_w = int(lr_w*self.scale[self.scale_idx])
        if self.bicubic_up:
            imgs_lq = [imresize(v, self.scale[self.scale_idx]) for v in imgs_lq]
            imgs_lq = torch.stack(imgs_lq, dim=0)
        result = {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (t, c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }
        result['gt'] = result['gt'][:, :, :hr_h, :hr_w]
        result['scale_idx'] = self.scale_idx
        return result
    def __len__(self):
        return len(self.data_info['gt_path'])