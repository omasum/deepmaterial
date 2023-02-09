import time
from turtle import shape
import numpy as np
import torch
from deepmaterial.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from scipy.spatial.transform import Rotation as R
from deepmaterial.utils import Render, random_tangent, reflect, set_random_seed, paraMirror
import torchvision
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
    
    set_random_seed(10)
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
    brdf_args['useAug']=False
    brdf_args['lampIntensity'] = 0.1
    brdf_args['dis'] = 1
    f = 5.08
    d = 5.08
    yOff = 10.16
    xy_range = {
        'x':[-d/2,d/2],
        'y':[yOff-d/2,yOff+d/2]
    }
    xy_reso = {
        'x': 256,
        'y': 256
    }
    T = [0, 0, 0]
    eular = [0, 0, 0]
    rotate_mode = 'xyz'
    mirror_args={
        'f': f,
        'dia': d,
        'y_offset': yOff,
        'xy_range': xy_range,
        'xy_reso': xy_reso,
        'rotation_mode': rotate_mode,
        'eular': eular,
        'T': T,
        'o': [1,10.16,10],
        'd': [0,0,-1],
        'fixed_light': True
    }
    # gamma_correct: drs
    opt['brdf_args'] = brdf_args
    opt['num_gpu'] = 1
    opt['mirror_args'] = mirror_args
    opt['phase'] = 'train'
    opt['name'] = 'brdf'
    opt['type'] = 'parabolicDataset'
    opt['data_root'] = '/home/sda/Dense3D'
    opt['len'] = 400000
    opt['range_d'] = [0,1]
    opt['range_s'] = [0,1]
    opt['range_r'] = [0.1,0.8]
    opt['spec_n'] = 20
    opt['mode'] = 'generate'
    opt['use_grad'] = True
    opt['io_backend']={'type':'disk'}
    # opt['range_n'] = {"theta":[0,70],"phi":[0,360]}

    # data loader
    opt['use_shuffle'] = False
    opt['num_worker_per_gpu'] = 4
    opt['batch_size_per_gpu'] = 32
    opt['dataset_enlarge_ratio'] = 1
    opt['pin_memory'] = True
    opt['prefetch_mode'] = 'cuda'
    
    # os.makedirs(f'tmp/{str(f)}-{str(d)}-{str(yOff)}', exist_ok=True)

    dataset_folder = build_dataset(opt)
    # opt['mode'] = 'generate'
    # dataset_generate = build_dataset(opt)
    dataset = dataset_folder
    train_sampler = EnlargedSampler(dataset, 1,
                                    0, 1)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, sampler=train_sampler)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    start_time=time.time()
    data = dataset.__getitem__(0)
    # dataset_generate = dataset_generate.__getitem__(0)
    # torchvision.utils.save_image(
    #     (data['measurements']**0.4545), f'tmp/Input1_folder.png', nrow=nrow, padding=padding, normalize=False)
    # torchvision.utils.save_image(
    #     (dataset_generate['measurements']**0.4545), f'tmp/Input1_generate.png', nrow=nrow, padding=padding, normalize=False)
    # torchvision.utils.save_image(
    #     (dataset_generate['trace'][6:]/2+0.5), f'tmp/wo_generate.png', nrow=nrow, padding=padding, normalize=False)
    # torchvision.utils.save_image(
    #     (data['trace'][6:]/2+0.5), f'tmp/wo_folder.png', nrow=nrow, padding=padding, normalize=False)
    # torchvision.utils.save_image(
    #         (dataset_generate['trace'][:3]), f'tmp/grad_x_generate.png', nrow=nrow, padding=padding, normalize=False)
    # torchvision.utils.save_image(
    #         (data['trace'][:3]), f'tmp/grad_x_folder.png', nrow=nrow, padding=padding, normalize=False)
    print("Per item need time: %f" % round(time.time()-start_time,2))
    del data
    def normalize(img):
        return (img-torch.min(img))/(torch.max(img)-torch.min(img))
    start_time=time.time()
    
    # dir_name = os.path.abspath(os.path.dirname(f'/home/hdr/disks/F/Datasets/Dense3D/{str(f)}-{str(d)}-{str(yOff)}/'))
    # os.makedirs(dir_name, exist_ok=True)
    prefetcher = CPUPrefetcher(data_loader)
    i = 0
    sub_dataset_num = 100
    # data_time=time.time()
    print('start...')
    for epoch in range(0, 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        data = prefetcher.next()
        # for idx, data in enumerate(data_loader):
        sub_dataset_OB = np.array([])
        sub_dataset_TBN = np.array([])
        sub_dataset_GT = np.array([])
        while data is not None:
            # data_time=time.time()-data_time
            # print('data_time', data_time)
            # time.sleep(1)
            if i >= 5:
                break
            # torch.save([data['TBN'][0].cpu(), data['brdf'][0].cpu()],f'tmp/TBN_GT_{i:07d}.pth')
            # if opt['mode'] == 'generate':
            #     if i == 0:
            #         sub_dataset_OB = (data['trace'][0]).unsqueeze(0).numpy().astype(np.uint8)
            #         sub_dataset_TBN = (data['TBN'][0]).unsqueeze(0).numpy()
            #         sub_dataset_GT = (data['brdf'][0]).unsqueeze(0).numpy()
            #     else:
            #         sub_dataset_OB = np.concatenate([sub_dataset_OB,(data['trace'][0]).unsqueeze(0).numpy().astype(np.uint8)], axis=0)
            #         sub_dataset_TBN = np.concatenate([sub_dataset_TBN,(data['TBN'][0]).unsqueeze(0).numpy()], axis=0)
            #         sub_dataset_GT = np.concatenate([sub_dataset_GT,(data['brdf'][0]).unsqueeze(0).numpy()], axis=0)
            #     if (sub_dataset_OB.shape[0]) % (opt['len']//sub_dataset_num) == 0:
            #         idx = (i+1) // (opt['len']//sub_dataset_num)
            #         np.save(f'/home/sda/Dense3D/{str(f)}-{str(d)}-{str(yOff)}/Input_{idx:03d}.npy',sub_dataset_OB)
            #         np.save(f'/home/sda/Dense3D/{str(f)}-{str(d)}-{str(yOff)}/TBN_{idx:03d}.npy',sub_dataset_TBN)
            #         np.save(f'/home/sda/Dense3D/{str(f)}-{str(d)}-{str(yOff)}/GT_{idx:03d}.npy',sub_dataset_GT)
            #         sub_dataset_OB = (data['trace'][0]).unsqueeze(0).numpy().astype(np.uint8)
            #         sub_dataset_TBN = (data['TBN'][0]).unsqueeze(0).numpy()
            #         sub_dataset_GT = (data['brdf'][0]).unsqueeze(0).numpy()
            #     # torchvision.utils.save_image(
            #     #     (data['measurements'][0]**0.4545), f'/home/sda/Dense3D/{str(f)}-{str(d)}-{str(yOff)}/Input_{i:07d}.png', nrow=nrow, padding=padding, normalize=False)
            #     # torch.save([data['TBN'][0], data['brdf'][0]],f'/home/sda/Dense3D/{str(f)}-{str(d)}-{str(yOff)}/TBN_GT_{i:07d}.pth')
            # else:
            #     torchvision.utils.save_image(
            #         (data['measurements']**0.4545), f'tmp/Measurements_{i:02d}.png', nrow=nrow, padding=padding, normalize=False)
            # render_result = (data['trace']-torch.min(data['trace']))/(torch.max(data['trace']) - torch.min(data['trace']))*data['mask']
            # print(torch.min(data['trace']), torch.max(data['trace']))
            render_result = data['trace']
            torchvision.utils.save_image(
                (data['measurements']**0.4545), f'tmp/Input_{i:07d}.png', nrow=nrow, padding=padding, normalize=False)
            svbrdfs = dataset.svbrdf.brdf2uint8(data['svbrdf'], n_xy=False)
            torchvision.utils.save_image(
                svbrdfs, f'tmp/brdf_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
            torchvision.utils.save_image(
                (render_result[:,6:])/2+0.5, f'tmp/wo_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
            # torchvision.utils.save_image(
            #     data['mask']*1.0, f'tmp/mask_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
            torchvision.utils.save_image(
                (data['noise']**0.4545), f'tmp/Noise_{i:07d}.png', nrow=nrow, padding=padding, normalize=False)
            torchvision.utils.save_image(
                (render_result[:,:3]), f'tmp/grad_x_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
            torchvision.utils.save_image(
                (render_result[:,3:6]), f'tmp/grad_y_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
            # torchvision.utils.save_image(
            #     (render_result), f'tmp/grad_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
            # torchvision.utils.save_image(
            #     data['spmask']*1.0, f'tmp/mask_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
            i+=1
            if (i) % 1000 == 0:
                print("Generated %d images, cost time: %f, still need time: %f hours." % ((i), round(time.time()-start_time,2)/3600, (time.time()-start_time)/i*(len(dataset)-i)/3600))
            # data_time=time.time()
            data = prefetcher.next()

            
    print("Test dataset done! Cost time: %f" % round(time.time()-start_time,2))


def main1(mode='folder'):
    """Test vimeo90k dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    set_random_seed(10)
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
    brdf_args['useAug']=False
    brdf_args['lampIntensity'] = 0.3
    yOff = 10.16
    d = 2.54
    f = 5.08
    renderer = Render(brdf_args) 
    xy_range = {
        'x':[-d/2,d/2],
        'y':[yOff-d/2,yOff+d/2]
    }
    xy_reso = {
        'x': 256,
        'y': 256
    }
    T = [0, 0, 0]
    eular = [0, 0, 0]
    rotate_mode = 'xyz'
    dis = 1

    mirror_args={
        'f': f,
        'dia': d,
        'y_offset': yOff,
        'xy_range': xy_range,
        'xy_reso': xy_reso,
        'rotation_mode': rotate_mode,
        'eular': eular,
        'T': T
    }

    mirror = paraMirror(opt = mirror_args)
    # mirror.plt_func()
    o = np.array([1,7,10])
    d = np.array([0,0,-1])
    
    x = mirror.intersect(o,d)[0]
    n = mirror.sample_dir(world=True)
    t = random_tangent(n)
    b = np.cross(n, t)
    TBN = np.stack([t,b,n], axis=1).T # TBN is the matrix that translate local vector to world vector.
    
    wi = torch.from_numpy(mirror.get_wi(o,d,TBN))
    # print(wi)
    wo, s, mask = mirror.get_wo(TBN)
    wos = torch.from_numpy(wo)

    wi_exp = wi.view(3,1,1).expand_as(wos).unsqueeze(0)

    ln = torch.from_numpy(np.matmul(TBN, n)).view(3,1,1)
    data = torch.from_numpy(np.array([0.1,0.5,0.3, 0.1, 0.1, 0.3, 0.5], dtype=np.float32))
    data = data*2-1

    svbrdf = torch.cat([ln.expand_as(wos), renderer.homo2sv(brdf=data)], axis=0)
    dis = torch.ones_like(wos)*dis
    wos.unsqueeze_(0)
    measurements, _, _, _, _ = renderer.render(svbrdf,light_dir=wi_exp, view_dir=wos, light_dis=dis)
    measurements *= mask
    
    nrow = 1
    padding = 0
    torchvision.utils.save_image(
        (measurements**0.4545), f'tmp/render2.png', nrow=nrow, padding=padding, normalize=False)
    
    # Simulate the random specular interreflectance
    mean_wo = np.mean(wo.reshape(3,-1),axis=-1)
    
    svbrdf = torch.cat([ln, data.view(7,1,1)], axis=0)
    Lo, _, _, _, _ = renderer.render(svbrdf,light_dir=wi.view(3,1,1), view_dir=torch.from_numpy(mean_wo).view(3,1,1), light_dis=torch.ones((3,1,1))*1)
    rwo = reflect(wo, n=np.tile(np.matmul(TBN, n).reshape(3,1,1),(1,256,256)), axis=0)
    mean_rwo = reflect(mean_wo, n=np.matmul(TBN, n))
    h = np.random.uniform(0,1, size=rwo.shape)
    Lr = np.random.beta(2,10, size=rwo.shape)

    rwo = rwo.reshape(3,-1)
    Lr = Lr.reshape(3,-1)
    idx = np.random.permutation(np.arange(rwo.shape[1]))
    rwo = rwo[:,idx[:200]]
    Lr = Lr[:,idx[:200]]*0.0001
    
    #----cosine noise-----
    # cosine_rwo = np.sum(rwo*mean_rwo.reshape(3,1),axis=0, keepdims=True)
    # Lr = np.sum(Lr[:,idx[:200]]*cosine_rwo, axis=1, keepdims=True)
    # Lr = Lo.numpy().reshape(3,-1)*np.exp(Lr)/np.sum(np.exp(Lr))
    # rwo = torch.from_numpy(mean_rwo.reshape(3,1))

    Lr = torch.from_numpy(Lr)
    rwo = torch.from_numpy(rwo)
    wo = torch.from_numpy(wo)
    noise = renderer.render_panel_single_point(svbrdf, Lr, rwo, None, wo)*mask
    ratio = np.random.uniform(0.01, 0.1)
    # noise *= ratio*Lo/torch.max(noise)

    # sir = Lr*np.sum(rwo*np.tile(mean_rwo.reshape(3,1,1),(1,256,256)), axis=0) 
    torchvision.utils.save_image(
        (noise**0.4545), f'tmp/noise.png', nrow=nrow, padding=padding, normalize=False)
    torchvision.utils.save_image(
        ((noise+measurements)**0.4545), f'tmp/total.png', nrow=nrow, padding=padding, normalize=False)
    
    print('Test done')



if __name__ == '__main__':
    main()
