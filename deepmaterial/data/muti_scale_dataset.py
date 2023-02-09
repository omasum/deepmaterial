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
class VideoTestVimeo90KDatasetMETA(VideoTestVimeo90KDataset):
    def __init__(self, opt):
        self.scale = opt['scale']
        if opt['useDir']:
            scale = float(opt['name'].split('-')[-1])
        super(VideoTestVimeo90KDatasetMETA,self).__init__(opt)
        if opt['useDir']:
            for i, lq in enumerate(self.data_info['lq_path']):
                root, subpath = lq[self.opt['num_frame'] // 2].split('x'+str(scale))
                self.data_info['lq_path'][i] = [root + 'x' + str(scale) + '/sequences' + subpath]

    def __getitem__(self, index):
        results = super(VideoTestVimeo90KDatasetMETA, self).__getitem__(index)
        results['lq'] = results['lq'][self.opt['num_frame']//2]
        results['lq'] = (results['lq']*255).int().float()
        results['gt'] = (results['gt']*255).int().float()
        H,W = results['lq'].shape[1:3]
        outH,outW = int(H*self.scale), int(W*self.scale)
        results['gt'] = results['gt'][:,0:outH,0:outW]
        return results

@DATASET_REGISTRY.register()
class Vimeo90KDatasetOTF(Vimeo90KDataset):
    def __init__(self, opt):
        self.scale = opt['scale']
        self.fix_hr = opt['fix_hr']
        self.scale_idx = 0
        super(Vimeo90KDatasetOTF,self).__init__(opt)
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
        # if self.is_lmdb:
        #     img_gt_path = f'{key}/im4'
        # else:
        #     img_gt_path = self.gt_root / clip / seq / 'im4.png'
        # img_bytes = self.file_client.get(img_gt_path, 'gt')
        # img_gt = imfrombytes(img_bytes, float32=False)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path_gt = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path_gt = self.gt_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path_gt, 'lq')
            img_lq_gt = imfrombytes(img_bytes, float32=False)
            img_lqs.append(img_lq_gt)

        #如果使用边缘信息当做输入，则数据集生成边缘信息
        if self.use_edge:
            img_lqs_edge = img2edge(img_lqs, keep_dims = True)
        else:
            img_lqs_edge = None

        img_lqs, img_lqs_edge = self.get_patch(img_lqs, img_lqs_edge)


        if self.opt['OTFMode'] == 'PIL':
            resizeFunc = imresize_PIL
        elif self.opt['OTFMode'] == 'CV':
            resizeFunc = imresize
        if self.fix_hr:
            hr_size = self.opt['lq_size']
            lr_size_init = lr_size = int(hr_size/scale)
            for i in range(lr_size_init):
                lr_size = lr_size_init-i
                if lr_size % 4 == 0:
                    break
            hr_size = int(lr_size*scale)
            img_gt = [l[:hr_size, :hr_size,:] for l in img_lqs]
        else:
            hr_size = self.opt['lq_size']
            lr_size = int(hr_size/scale)
            hr_size = int(lr_size*scale)
            img_gt = [l[:hr_size, :hr_size,:] for l in img_lqs]
        img_lqs = resizeFunc(img_gt, size=(lr_size,lr_size))
            # lr_size = 64
            # img_gt = resizeFunc(img_lqs, size=(int(lr_size*scale),int(lr_size*scale)))
            # img_lqs = resizeFunc(img_gt, size=(lr_size,lr_size))
        if self.use_edge:
            img_lqs_edge = resizeFunc(img_lqs_edge, size=(lr_size,lr_size))
        img_gt = img_gt[self.opt['num_frame'] // 2]
        
        img_lqs = img2tensor(img_lqs, normalization=True)
        img_gt = img2tensor(img_gt, normalization=True)

        img_lqs = torch.stack(img_lqs, dim=0)
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

    def get_patch(self, img_lq, img_lq_edge=None):
        scale = self.scale[self.scale_idx]
        # multi_scale = len(self.scale) > 1
        if self.opt['phase']=='train':
            lq_patch_size = self.opt['lq_size']
            lq_length = len(img_lq)
            results = paired_random_crop_mutiscale(
                img_lq,
                img_lqs_edge=img_lq_edge,
                patch_size=lq_patch_size,
                scale=scale
            )
            
            if img_lq_edge is not None:
                if lq_length != 1:
                    img_lq, img_lq_edge = results
                    img_lq.extend(img_lq_edge)
                else:
                    img_lq = [img_lq, img_lq_edge]
            else:
                img_lq = results
                
            # img_lq = augment(img_lq, hflip = self.opt['use_flip'],
            #                         rotation = self.opt['use_rot'])
            
            if img_lq_edge is not None:
                img_lq_edge = img_lq[lq_length:2*lq_length]
            if lq_length != 1:
                img_lq = img_lq[0:lq_length]
            else:
                img_lq = [img_lq]
        else:
            if not isinstance(img_lq, list):
                img_lq = [img_lq]
            if img_lq_edge is not None and not isinstance(img_lq_edge, list):
                img_lq_edge = [img_lq_edge]
            ih, iw = img_lq.shape[:2]

        results = [img_lq, img_lq_edge]

        return results

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
    def get_scale(self):
        return self.scale[self.scale_idx]

@DATASET_REGISTRY.register()
class Vimeo90KDatasetOTFTest(data.Dataset):
    def __init__(self, opt):
        self.scale = opt['scale']
        if 'fixed_scale' in opt:
            self.scale_idx = self.scale.index(opt['fixed_scale'])
        else:
            self.scale_idx = 0
        super(Vimeo90KDatasetOTFTest, self).__init__()
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
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)
            lq_path = [
                osp.join(self.gt_root, subfolder, f'im{i}.png')
                for i in neighbor_list
            ]
            self.data_info['lq_path'].append(lq_path)

            self.data_info['folder'].append('vimeo90k')
            self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
            self.data_info['border'].append(0)

    def __getitem__(self, index):
        scale = self.get_scale()
        lq_path = self.data_info['lq_path'][index]
        gt_path = self.data_info['gt_path'][index]
        imgs_lqs = read_img_seq(lq_path, numpy=True)
        hr_h, hr_w = imgs_lqs[0].shape[:2]
        if self.opt["fix_hr"]:
            lr_h_init = lr_h = int(hr_h/scale)
            for i in range(lr_h_init):
                lr_h = lr_h_init-i
                if lr_h % 4 == 0 and int(lr_h * (hr_w/hr_h)) % 4 == 0:
                    break
            lr_w = int(lr_h * (hr_w/hr_h))
            # lr_w_init = lr_w = int(hr_w/scale)
            # for i in range(lr_w_init):
            #     lr_w = lr_w_init-i
            #     if lr_w % 4 == 0:
            #         break
        else:
            lr_h = int(hr_h/scale)
            lr_w = int(hr_w/scale)
        hr_h = int(lr_h*scale)
        hr_w = int(lr_w*scale)
        img_gt = [l[:hr_h, :hr_w,:] for l in imgs_lqs]

        if self.opt['OTFMode'] == 'PIL':
            resizeFunc = imresize_PIL
        elif self.opt['OTFMode'] == 'CV':
            resizeFunc = imresize
        imgs_lqs = resizeFunc(img_gt, size=(lr_h,lr_w))
        if self.use_edge:
            img_lqs_edge = img2edge(imgs_lqs, keep_dims = True)
            img_lqs_edge = resizeFunc(img_lqs_edge, size=(lr_h,lr_w))
            
        img_gt = img_gt[self.opt['num_frame'] // 2]
        img_gt = img2tensor(img_gt, normalization=True)
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
class PairedImageDatasetMETATest(PairedImageDataset):
    def __init__(self, opt):
        self.scale = opt['scale']
        self.scale_idx = self.scale.index(opt['fixed_scale'])
        super(PairedImageDatasetMETATest,self).__init__(opt)

    def __getitem__(self, index):
        results = super(PairedImageDatasetMETATest,self).__getitem__(index)
        results['lq'] = (results['lq']).int().float()
        results['gt'] = (results['gt']).int().float()
        H,W = results['lq'].shape[1:3]
        outH,outW = int(H*self.scale[self.scale_idx]), int(W*self.scale[self.scale_idx])
        results['gt'] = results['gt'][:,0:outH,0:outW]
        results['scale_idx'] = self.scale_idx
        return results

@DATASET_REGISTRY.register()
class PairedImageDatasetMETA(data.Dataset):
    def __init__(self, opt):
        super(PairedImageDatasetMETA,self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        self.paths = muti_scale_paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl
        )
        self.scale = opt['scale']
        self.scale_idx = 0
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.scale[self.scale_idx]

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path'][scale]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_gt, img_lq = self.get_patch(img_gt, img_lq)

        # color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        results = {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

        # results = self.mutiScaleDatasets[str(self.scale)].__getitem__(index)
        results['lq'] = (results['lq']*255).int().float()
        results['gt'] = (results['gt']*255).int().float()
        # H,W = results['lq'].shape[1:3]
        # outH,outW = int(H*self.scale), int(W*self.scale)
        # results['gt'] = results['gt'][:,0:outH,0:outW]
        return results

    def get_patch(self, img_gt, img_lq):
        scale = self.scale[self.scale_idx]
        # multi_scale = len(self.scale) > 1
        if self.opt['phase']=='train':
            lq_patch_size = self.opt['lq_size']
            results = paired_random_crop_mutiscale(
                img_gt,
                img_lq,
                patch_size=lq_patch_size,
                scale=scale
            )
            img_gt, img_lq = results
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])
        else:
            ih, iw = img_lq.shape[:2]
            img_gt = img_gt[0:int(ih * scale), 0:int(iw * scale)]

        return img_gt, img_lq
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
    def get_scale(self):
        return self.scale[self.scale_idx]
        
    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class msVimeo90KDataset(Vimeo90KDataset):
    def __init__(self, opt):
        self.scale = opt['scale']
        self.fix_hr = opt['fix_hr']
        self.scale_idx = -1
        super(msVimeo90KDataset,self).__init__(opt)
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
        if self.is_lmdb:
            img_gt_path = f'{key}/im4'
        else:
            img_gt_path = self.gt_root / clip / seq / 'im4.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=False)

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

        img_gt, img_lqs, img_lqs_edge = self.get_patch(img_lqs, img_gt, img_lqs_edge)
        
        img_lqs = img2tensor(img_lqs, normalization=True)
        img_gt = img2tensor(img_gt, normalization=True)

        img_lqs = torch.stack(img_lqs, dim=0)
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
                img_lq.append(img_gt)
                
            img_lq = augment(img_lq, hflip = self.opt['use_flip'],
                                    rotation = self.opt['use_rot'])
            
            if img_lq_edge is not None:
                img_lq_edge = img_lq[lq_length:2*lq_length]
                img_gt = img_lq[2*lq_length]
            else:
                img_gt = img_lq[lq_length]
            img_lq = img_lq[0:lq_length]

        results = [img_gt, img_lq, img_lq_edge]

        return results

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
    def get_scale(self):
        return self.scale[self.scale_idx]


@DATASET_REGISTRY.register()
class msVimeo90KDatasetTest(data.Dataset):
    def __init__(self, opt):
        self.scale = opt['scale']
        if 'fixed_scale' in opt:
            self.scale_idx = self.scale.index(opt['fixed_scale'])
        else:
            self.scale_idx = 0
        super(msVimeo90KDatasetTest, self).__init__()
        self.opt = opt
        self.use_edge = True if 'use_edge' in self.opt and self.opt['use_edge'] else False
        self.cache_data = opt['cache_data']
        self.bicubic_up = opt.get('bicubic_up',False)
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
            gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)
            lq_path = [
                # osp.join(self.lq_root, subfolder, f'im{i}.png')
                self.lq_root+"/"+subfolder+f'/im{i}.png'
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


        # for i in range(6):
        #     tmp = lq_path.pop(0)
        #     lq_path.append(tmp)
        #     if i < 2:
        #         idx = (5+i)%7
        #     else:
        #         idx = i-2
        #     gt_path = gt_path.replace('im4', 'im'+str(idx))

        imgs_lqs = read_img_seq(lq_path, numpy=True)
        img_gt = read_img_seq([gt_path], numpy=True)
        hr_h, hr_w = img_gt[0].shape[:2]
        lr_h, lr_w,_ = imgs_lqs[0].shape
        hr_h = int(lr_h*scale)
        hr_w = int(lr_w*scale)
        img_gt = img_gt[0][:hr_h, :hr_w,:]

        if self.use_edge:
            img_lqs_edge = img2edge(imgs_lqs, keep_dims = True)
            
        img_gt = img2tensor(img_gt, normalization=True)
        imgs_lqs = torch.stack(img2tensor(imgs_lqs, normalization=True),dim=0)
        if self.bicubic_up:
            # print("upsample lq with scale %.2f"%self.scale[self.scale_idx])
            imgs_lqs = [imresize(v, self.scale[self.scale_idx]) for v in imgs_lqs]
            imgs_lqs = torch.stack(imgs_lqs, dim=0)
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
class msVid4(data.Dataset):
    def __init__(self, opt):
        super(msVid4, self).__init__()
        self.opt = opt
        self.scale=opt['scale']
        if isinstance(self.scale,list):
            self.scale_idx = self.scale.index(opt['fixed_scale'])
        else:
            self.scale_idx = -1
        self.cache_data = opt['cache_data']
        self.bicubic_up = opt.get('bicubic_up',False)
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'
        self.mode_base = opt.get('mod_base',None)
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

        if opt['name'].lower() in ['vid4','msvid4', 'reds4', 'redsofficial']:
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
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        if self.bicubic_up:
            imgs_lq = [imresize(v, self.scale[self.scale_idx]) for v in imgs_lq]
            imgs_lq = torch.stack(imgs_lq, dim=0)
        if self.mode_base:
            t,c,h,w = imgs_lq.shape
            mod_h = h
            for i in range(h):
                mod_h = mod_h-i
                if mod_h % self.mode_base == 0:
                    break
            mod_w = w
            for i in range(w):
                mod_w = mod_w-i
                if mod_w % self.mode_base == 0:
                    break
            imgs_lq = imgs_lq[:,:,:mod_h, :mod_w]
            if self.bicubic_up:
                img_gt = img_gt[:,:mod_h, :mod_w]
            else:
                img_gt = img_gt[:,:int(mod_h*self.scale[self.scale_idx]), :int(mod_w*self.scale[self.scale_idx])]
        result = {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }
        _, lr_h, lr_w = result['lq'][0].shape
        if isinstance(self.scale,list):
            scale = self.scale[self.scale_idx]
        else:
            scale = self.scale
        hr_h = int(lr_h*scale)
        hr_w = int(lr_w*scale)
        result['gt'] = result['gt'][:, :hr_h, :hr_w]
        result['scale_idx'] = self.scale_idx
        return result
    def __len__(self):
        return len(self.data_info['gt_path'])