from copy import deepcopy
import os
from time import time
import numpy as np
from torch.utils import data as data
import torch, random

from deepmaterial.data.data_util import paths_from_folder
from deepmaterial.utils import FileClient, imfrombytes, img2tensor, random_tangent\
    , torch_dot, torch_norm, reflect, finit_difference_uv, paraMirror,\
    preprocess, toHDR_torch, toLDR_torch, log_normalization, imresize, Render, PlanarSVBRDF, Lighting
from deepmaterial.utils.registry import DATASET_REGISTRY
from deepmaterial.utils.render_util import PolyRender, RectLighting


@DATASET_REGISTRY.register()
class svbrdfDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read svBRDFs (material for rendering).

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_train (str): Data root path for train svBRDFs.
            dataroot_test (str): Data root path for test svBRDFs.
            io_backend (dict): IO backend type and other kwarg.
            input_type (str): 'brdf' or 'img'
            gt_size (int): Cropped patched size for gt patches.
            split_num (int): number to split read image
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).
            use_ambient (bool): Use ambient light rendering
            jitterlight (bool): Jitter the light and view direction
            
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(svbrdfDataset, self).__init__()
        self.opt = opt
        # self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        data_folder = opt['data_path']
        
        self.data_paths = sorted(paths_from_folder(data_folder))
        # self.svBRDF_utils = svBRDF(self.opt['brdf_args'])
        self.renderer = Render(self.opt['brdf_args'])
        
        self.svbrdf = PlanarSVBRDF(self.opt['brdf_args'])
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        if self.opt.get('fixed_input', False):
            img_bytes = self.file_client.get(self.opt['fixed_path'], 'brdf')
            img = imfrombytes(img_bytes, float32=True, bgr2rgb=True)
            self.svbrdf.get_svbrdfs(img)
            self.inputs = self.renderer.render(self.svbrdf, keep_dirs=True)
        # self.sampler = torch.distributions.Uniform(-1.0, 1.0)

    def __getitem__(self, index):

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_path = self.data_paths[index]
        img_bytes = self.file_client.get(img_path, 'brdf')
        img = imfrombytes(img_bytes, float32=True, bgr2rgb=True)
        
        if self.opt['input_type'] == 'brdf':
            self.svbrdf.get_svbrdfs(img)
            if not self.opt.get('fixed_input', False):
                inputs = self.renderer.render(self.svbrdf, keep_dirs=True)
            else:
                inputs = self.inputs
            light_pos, view_pos, surface, distance = self.renderer.light_dir, self.renderer.light_dir, self.renderer.surface, self.renderer.light_dis
            self.svbrdf.squeeze_brdf()
        # augmentation rendering for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        return {'inputs': inputs, 'svbrdfs': self.svbrdf.brdf, 'path': img_path, 'light':light_pos, 'view':view_pos, 'surface':surface, 'dis': distance}

    def __len__(self):
        return len(self.data_paths)

@DATASET_REGISTRY.register()
class normalDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read svBRDFs (material for rendering).

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_train (str): Data root path for train svBRDFs.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(normalDataset, self).__init__()
        self.opt = opt
        # self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.train_folder = opt['dataroot_train']
        
        self.train_paths = paths_from_folder(self.train_folder)
        # self.test_paths = paths_from_folder(self.test_folder)
        self.svBRDF = PlanarSVBRDF(self.opt['brdf_args'])
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # self.sampler = torch.distributions.Uniform(-1.0, 1.0)

    def __getitem__(self, index):

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_path = self.train_paths[index]
        img_bytes = self.file_client.get(img_path, 'brdf')
        img = imfrombytes(img_bytes, float32=True, bgr2rgb=True)
        
        self.svBRDF.get_svbrdfs(img)
        return {'svbrdfs': self.svBRDF.brdf, 'path': img_path, 'normal': self.svBRDF.brdf[:3]}

    def __len__(self):
        return len(self.train_paths)

@DATASET_REGISTRY.register()
class homoDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.renderer = Render(self.opt['brdf_args'])
        
        self.rand_nb = self.opt.get('rand_nb', False)
        self.nb=self.opt['nb']
        self.rand_lv = self.opt.get('rand_lv', False)
        self.svbrdf = PlanarSVBRDF(self.opt['brdf_args'])
        if not self.rand_lv:
            self.l = self.renderer.lighting.fixedLightsSurface()
            self.v = self.renderer.camera.fixedView()
        
    def set_nb(self, nb):
        self.nb = nb

    def __getitem1__(self, index):
        # generate random d, r, s in range from opt
        range_d = self.opt['range_d']
        range_r = self.opt['range_r']
        range_s = self.opt['range_s']
        if self.opt['phase']=='train' and self.rand_nb:
            nb = self.opt['nb']
            num = np.random.randint(1, nb)
        else:
            nb = self.nb
            num = self.nb
        s = time()
        data = torch.rand((7,), dtype=torch.float32)
        data[0:3] = data[0:3]*(range_d[1]-range_d[0])+range_d[0]
        data[3:4] = data[3:4]*(range_r[1]-range_r[0])+range_r[0]
        data[4:7] = data[4:7]*(range_s[1]-range_s[0])+range_s[0]
        data = data*2-1
        n, spmask = self.renderer.sphere_normal(padding=True)
        # print('generate normal: ', time()-s)
        # n = self.renderer.plane_normal()
        svbrdf = PlanarSVBRDF.homo2sv(data, n.shape[-2:])
        brdf = torch.cat([n,svbrdf], dim=0)
        self.svbrdf.setSVBRDF(brdf)
        inputs = self.renderer.render(self.svbrdf, keep_dirs=True)
        light_dir, view_dir, obj_pos, distance = self.renderer.light_dir, self.renderer.light_dir, self.renderer.surface, self.renderer.light_dis

        spmask = (torch_dot(n,light_dir)<0)
        # print('render: ', time()-s)

        trace, mask = self.renderer.shuffle_sample(nb, num, n, inputs, light_dir, view_dir, obj_pos, distance, spmask)
        trace = torch.from_numpy(trace)
        mask = torch.from_numpy(mask.astype(np.bool8))
        results = {
            'trace': trace,
            'brdf' : data,
            'mask' : mask,
            'inputs': inputs,
            'spmask': spmask,
            'svbrdf': brdf
        }
        # print('shuffle and translate: ', time()-s)
        return results
    
    def __getitem__(self, index):
        range_d = self.opt['range_d']
        range_r = self.opt['range_r']
        range_s = self.opt['range_s']
        range_n = self.opt['range_n']
        theta_range = self.opt['range_n']['theta']
        phi_range = self.opt['range_n']['phi']
        if self.opt['phase']=='train' and self.rand_nb:
            nb = self.opt['nb']
            num = np.random.randint(10, self.opt['nb'])
        else:
            nb = self.opt['nb']
            num = self.nb
        data = torch.rand((7,1), dtype=torch.float32)
        data[0:3] = data[0:3]*(range_d[1]-range_d[0])+range_d[0]
        data[3:4] = data[3:4]*(range_r[1]-range_r[0])+range_r[0]
        data[4:7] = data[4:7]*(range_s[1]-range_s[0])+range_s[0]
        data = data*2-1
        if self.opt.get('mix_plane', False):
            rate = self.opt.get('plane_rate', 0.3)
            pnum = num-int(num*rate)
            n = self.renderer.dirUtil.random_dir(n = num-pnum, theta_range=theta_range, phi_range=phi_range)
            theta_range = [0, 20]
            pn = self.renderer.dirUtil.random_dir(n = 1, theta_range=theta_range, phi_range=phi_range).repeat(1, pnum)
            n = torch.cat([n,pn],dim=1)
        else:
            n = self.renderer.dirUtil.random_dir(n = num, theta_range=theta_range, phi_range=phi_range)
        # n = self.renderer.plane_normal().view(3,-1)[:,:num]
        brdf = torch.cat([n,data.repeat(1,num)], dim=0)
        x = self.renderer.dirUtil.random_pos(n = num).unsqueeze(0)
        if self.rand_lv:
            self.l = self.renderer.lighting.fullRandomLightsSurface()
            self.v = self.l
        if self.opt.get('jitter', False):
            jit = torch.empty_like(self.v).normal_(std=0.1)
            l_jit = (self.l + jit).unsqueeze(-1)
            v_jit = (self.v + jit).unsqueeze(-1)
        else:
            l_jit = self.l.unsqueeze(-1)
            v_jit = self.v.unsqueeze(-1)
        l_dir=torch_norm(l_jit-x, dim=-2)
        v_dir=torch_norm(v_jit-x, dim=-2)
        l_dis=torch.sum(torch.square(l_jit - x),-2,keepdims=True)
        ob = self.renderer.render(brdf, obj_pos=x, light_dir=l_dir, view_dir=v_dir, light_dis=l_dis, perPoint=True)
        trace = torch.cat([n,ob,l_dir.squeeze(0),v_dir.squeeze(0),x.squeeze(0)[:2],l_dis.squeeze(0)]).permute(1,0).contiguous()
        if self.opt.get('prefetch_mode', '') == 'cuda' or self.rand_nb:
            fill = torch.zeros((nb-trace.shape[0], 15), dtype=torch.float32)
            mask = torch.cat([torch.ones((trace.shape[0]), dtype=torch.bool),torch.zeros((nb-trace.shape[0]), dtype=torch.bool)],dim=0)
            trace = torch.cat([trace, fill],dim=0)
            results = {
                'trace': trace,
                'brdf' : data.squeeze(-1),
                'mask' : mask
                # 'inputs': inputs,
                # 'spmask': spmask,
                # 'svbrdf': brdf
            }
        else:
            results = {
                'trace': trace,
                'brdf' : data.squeeze(-1)
            }
        return results
        
    def __len__(self):
        return self.opt['len']


@DATASET_REGISTRY.register()
class parabolicDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dis = self.opt['brdf_args'].pop('dis')
        self.renderer = Render(self.opt['brdf_args'])
        self.svbrdf = PlanarSVBRDF(self.opt['brdf_args'])
        self.mirror = paraMirror(self.opt['mirror_args'])
        n = torch.from_numpy(np.array([0,0,-1],dtype=np.float32)).view(3,1,1)
        self.TBN = np.array([[1,0,0],[0,1,0],[0,0,-1]],dtype=np.float32)
        self.spec_n = self.opt['spec_n']
        self.mode= self.opt['mode']
        self.phase = self.opt['phase']
        self.use_grad = self.opt.get('use_grad', False)
        if self.mode == 'folder':
            self.file_client = None
            self.io_backend_opt = opt['io_backend']
            self.train_folder = opt['data_root']+f'/{self.mirror.focallength}-{self.mirror.dia}-{self.mirror.yoff}'
            paths = paths_from_folder(self.train_folder, suffix='npy')
            self.train_paths = paths
            self.obs = None
            self.GTs = None
            self.TBNs = None
            for path in sorted(paths):
                if 'Input' in path:
                    if self.obs is None:
                        self.obs = np.load(path)
                    else:
                        self.obs = np.concatenate([self.obs, np.load(path)], axis=0)
                elif 'TBN' in path:
                    if self.TBNs is None:
                        self.TBNs = np.load(path)
                    else:
                        self.TBNs = np.concatenate([self.TBNs, np.load(path)], axis=0)
                elif 'GT' in path:
                    if self.GTs is None:
                        self.GTs = np.load(path)
                    else:
                        self.GTs = np.concatenate([self.GTs, np.load(path)], axis=0)

            self.train_paths = self.obs

            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        # f = self.opt['mirror_args']['f']
        # self.xr = [-2*np.sqrt(f)+0.05,2*np.sqrt(f)-0.05-3]
        # self.yr = [-2*np.sqrt(f)+0.05,2*np.sqrt(f)-0.05-3]

        
    def __getitem__(self, index):
        if self.mode=='generate':
            range_d = self.opt['range_d']
            range_r = self.opt['range_r']
            range_s = self.opt['range_s']
            
            data = torch.rand((7,1,1), dtype=torch.float32)
            data[0:3] = data[0:3]*(range_d[1]-range_d[0])+range_d[0]
            data[3:4] = data[3:4]*(range_r[1]-range_r[0])+range_r[0]
            data[4:7] = data[4:7]*(range_s[1]-range_s[0])+range_s[0]
            data = data*2-1

            # sample the normal in the outgoing range, the tangent vector is a random vectore that perpendicular to normal        
            n = self.mirror.sample_dir(world=True)
            # n = torch.from_numpy(np.array([0,0, -1],dtype=np.float32))
            t = random_tangent(n)
            b = np.cross(n, t)
            TBN = np.stack([t,b,n], axis=1).T.astype(np.float32) # TBN is the matrix that translate local vector to world vector.

            if self.opt.get('jitter', False):
                ln = torch.from_numpy(np.array([0,0, 1],dtype=np.float32)).view(3,1,1) 
                ln = torch_norm(ln + torch.empty_like(ln).normal_(0, 0.02), dim=0)
                
            else:
                # The light/view direction has been rotated into local coordinate, hence the normal is [0,0,1]
                ln = torch.from_numpy(np.array([0,0, 1],dtype=np.float32)).view(3,1,1) 
                
            # generate the outgoing ray range and incident ray
            wo, s, mask = self.mirror.get_wo(TBN=TBN)
            wo = wo.astype(np.float32)
            mask = torch.from_numpy(mask)
            wwo = torch.from_numpy(wo).unsqueeze(0)
            wi = self.mirror.get_wi(TBN=TBN)
            wwi = torch.from_numpy(wi.astype(np.float32)).view(1,3,1,1).expand_as(wwo)

            # render the measurements captured by parabolic mirror
            brdf = torch.cat([ln,data], dim=0)
            svbrdf = brdf.repeat(1, *wwo.shape[-2:])
            self.svbrdf.setSVBRDF(svbrdf)
            l_dis = torch.ones_like(wwo)*self.dis**2
            l_dir=wwi
            v_dir=wwo
            ob = self.renderer.render(self.svbrdf, light_dir=l_dir, view_dir=v_dir, light_dis=l_dis)
            ob = torch.clip(ob, 0, 1)

            # use Beta(2,10) distribution to simulate specular noise
            idx = np.random.permutation(np.arange(wo.shape[1]*wo.shape[2]))
            nb = np.random.randint(0, self.spec_n)
            # nb=1
            
            if nb != 0:
                rwo = reflect(wo.reshape(3,-1)[:,idx[:nb]], n=np.tile(ln.reshape(3,1),(1,nb)), axis=0)
                Lr = np.random.beta(2,5, size=rwo.shape).astype(np.float32)
                self.svbrdf.setSVBRDF(brdf)
                noise = self.renderer.render_panel_single_point(self.svbrdf, torch.from_numpy(Lr), torch.from_numpy(rwo), None, torch.from_numpy(wo))
                ratio = np.random.uniform(0.01, 0.1)
                noise *= ratio*torch.max(ob)/torch.max(noise)
                ob = ob+noise
            results = {
                'trace': preprocess(ob)*mask,
                'brdf' : data.view(-1).float(),
                'measurements': ob*mask,
                'TBN': TBN,
                # 'noise':noise,
                # 'svbrdf':svbrdf
            }
            if self.use_grad:
                # calculate the finit difference of ob_n and wo for simulate the gradient.
                wo_diff_u, wo_diff_v = finit_difference_uv(torch.from_numpy(wo))
                ob_diff_u, ob_diff_v = finit_difference_uv(ob)
                mindiff = torch.abs(wo_diff_u)>torch.abs(wo_diff_v)

                # calculate gradient from ob_n to wo(x,y)
                grad_x = (torch.nan_to_num(ob_diff_u/wo_diff_u[0:1]*mindiff[0:1]) + torch.nan_to_num(ob_diff_v/wo_diff_v[0:1]*(~mindiff[0:1])))
                grad_y = (torch.nan_to_num(ob_diff_u/wo_diff_u[1:2]*mindiff[1:2]) + torch.nan_to_num(ob_diff_v/wo_diff_v[1:2]*(~mindiff[1:2])))
                grad = torch.concat([grad_x,grad_y, wwo.squeeze(0)[:,:-1,:-1]], dim=0)*mask[:-1,:-1]
                results['trace'] = grad.float()
        elif self.mode == 'folder':
            # start = time()
            # p = self.train_paths[index]
            # ob_path = p
            # gt_tbn_path = p.replace('Input','TBN_GT')
            # gt_tbn_path = gt_tbn_path.replace('png','pth')

            brdf = torch.from_numpy(self.GTs[index])
            # img_bytes = self.file_client.get(ob_path, 'brdf')
            # img = imfrombytes(img_bytes, float32=False)
            # ob = img2tensor(img, normalization=False)
            ob = torch.from_numpy(self.obs[index])
            ob = toHDR_torch(ob)
            # print('reading' , time()-start)
            TBN = self.TBNs[index]
            wo, s, mask = self.mirror.get_wo(TBN=TBN)
            
            results = {
                'trace': preprocess(ob)*mask,
                'brdf' : brdf,
                # 'measurements': ob,
            }
            if self.use_grad:
                wwo = torch.from_numpy(wo).unsqueeze(0)
                
                # calculate the finit difference of ob_n and wo for simulate the gradient.
                wo_diff_u, wo_diff_v = finit_difference_uv(torch.from_numpy(wo))
                ob_diff_u, ob_diff_v = finit_difference_uv(ob)
                mindiff = torch.abs(wo_diff_u)>torch.abs(wo_diff_v)

                # calculate gradient from ob_n to wo(x,y)
                grad_x = (torch.nan_to_num(ob_diff_u/wo_diff_u[0:1]*mindiff[0:1]) + torch.nan_to_num(ob_diff_v/wo_diff_v[0:1]*(~mindiff[0:1])))
                grad_y = (torch.nan_to_num(ob_diff_u/wo_diff_u[1:2]*mindiff[1:2]) + torch.nan_to_num(ob_diff_v/wo_diff_v[1:2]*(~mindiff[1:2])))
                grad = torch.concat([grad_x,grad_y, wwo.squeeze(0)[:,:-1,:-1]], dim=0)*mask[:-1,:-1]
                results['trace'] = grad.float()
        return results
        
    def __len__(self):
        if self.mode=='generate':
            return self.opt['len']
        else:
            return self.obs.shape[0]

@DATASET_REGISTRY.register()
class SurfaceNetDataset(data.Dataset):
    def __init__(self,opt):
        super(SurfaceNetDataset, self).__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        self.svbrdf_folder = opt['svbrdf_root']
        self.svbrdf_paths = paths_from_folder(self.svbrdf_folder)
        self.svbrdf = PlanarSVBRDF(self.opt['brdf_args'])
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
    def __getitem__(self, index):
    
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        img_path = self.svbrdf_paths[index]
        img_bytes = self.file_client.get(img_path, 'brdf')
        img = imfrombytes(img_bytes, float32=True)[:,:,::-1]
        inputs = img[:,:256]
        inputs = img2tensor(inputs.copy(),bgr2rgb=False)
        svbrdfs = self.svbrdf.get_svbrdfs(img)
        return {'inputs': inputs*2-1, 'svbrdfs': svbrdfs, 'name':os.path.basename(self.svbrdf_paths[index])}
    def __len__(self):
        return len(self.svbrdf_paths)

@DATASET_REGISTRY.register()
class areaDataset(svbrdfDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self.light_mode=self.opt.get('light_mode', 'area')
        self.input_mode=self.opt.get('input_mode', 'render') # 'folder', 'render', 'image'
        if not self.opt.get('worker_init', False):
            self.init()

    def init(self):
        rendering_folder = self.opt.get('rendering_path', None)
        if self.input_mode == 'render':
            if self.light_mode == 'area':
                self.light_config = self.opt.get('light_args', {'type':'RectLighting'})
                self.light_config.update(self.opt['brdf_args'])
                tex = self.light_config.get('texture', '')
                if isinstance(tex, list):
                    self.lighting = []
                    self.texture = []
                    config = deepcopy(self.light_config)
                    for t in tex:
                        config['texture'] = t
                        lighting = RectLighting(config)
                        self.lighting.append(lighting)
                        lh, lw = lighting.tex.shape[-2:]
                        self.texture.append(imresize(lighting.tex,scale=256/lh))
                    self.renderer = PolyRender(self.opt['brdf_args'])
                else:
                    self.lighting = RectLighting(self.light_config)
                    self.renderer = PolyRender(self.opt['brdf_args'], lighting=self.lighting)
                    lh, lw = self.renderer.lighting.tex.shape[-2:]
                    self.texture = imresize(self.renderer.lighting.tex,scale=256/lh)
        
            elif self.light_mode == 'parallel':
                self.renderer = Render(self.opt['brdf_args'])
                self.light_dir = self.opt['brdf_args']['lightdir']
        elif self.input_mode == 'folder':
            if self.opt['phase'] == 'train':
                input_folder = os.path.join(rendering_folder, 'train-'+self.light_mode+'Lighting-large')
            else:
                input_folder = os.path.join(rendering_folder, 'test-'+self.light_mode+'Lighting-large')
            self.input_path = sorted(paths_from_folder(input_folder))
        else:
            self.input_path = None
        self.data_paths = self.data_paths

    def __getitem__(self, index):
        img_path = self.data_paths[index]
        img_bytes = self.file_client.get(img_path, 'brdf')
        svbrdf_img = imfrombytes(img_bytes, float32=True, bgr2rgb=True) # original iamges/255: 0-1
        pattern = {}
        h,w,c = svbrdf_img.shape
        # svbrdf_img = imresize(svbrdf_img, scale = 256/288) # 2080ti
        svbrdfs = self.svbrdf.get_svbrdfs(svbrdf_img)
        svbrdfs = torch.clip(svbrdfs, -1.0, 1.0)
        # svbrdfs = self.svBRDF_utils.unsqueeze_brdf(svbrdfs, n_xy=False, r_single=True)
        if self.input_mode == 'folder':
            input_path = self.input_path[index]
            img_bytes = self.file_client.get(input_path, 'brdf')
            inputs_img = imfrombytes(img_bytes, float32=False, bgr2rgb=False)
            inputs_img = img2tensor(inputs_img, bgr2rgb=True, float32=True, normalization=True)
        elif self.input_mode == 'image':
            inputs_img = img2tensor(svbrdf_img[:h, :h], bgr2rgb=False)
            inputs = torch.clip(inputs_img, 0.0, 1.0)
        else:
            if self.light_mode == 'area':
                if isinstance(self.lighting, list):
                    inputs = []
                    lightingInputs = []
                    for i, lighting in enumerate(self.lighting):
                        self.renderer.lighting = lighting
                        tmpLight = self.texture[i]
                        lightingInputs.append(tmpLight)
                        inputs.append(self.renderer.render(svbrdf=svbrdfs))
                    inputs_img = torch.stack(inputs, dim=0) ** 0.4545
                    inputs.extend(lightingInputs)
                    inputs = torch.cat(inputs, dim=-3)
                else:
                    inputs = self.renderer.render(svbrdf=svbrdfs)
                    inputs_img = inputs ** 0.4545
                    if self.opt.get('catTex', False):
                        inputs = torch.cat([inputs, self.texture], dim=-3)
                    pattern['pattern'] = self.renderer.lighting.tex*2-1
            else:
                if self.light_mode == 'point':
                    inputs = self.renderer.render(svbrdf=svbrdfs, random_light=False)
                    inputs_img = inputs ** 0.4545
                else: # parallel
                    inputs = self.renderer.render(svbrdf=svbrdfs, random_light=False, light_dir = torch.tensor(self.light_dir)) # no gamma
                    inputs_img = inputs ** 0.4545
                
        if not self.opt.get('gamma', True):
            inputs = inputs ** 0.4545

        if self.opt.get('log', False):
            inputs = log_normalization(inputs)
        inputs = preprocess(inputs)
        result = {
            'inputs': inputs, # input of net, without gamma
            'imgs': inputs_img, # show, with gamma
            'svbrdfs': svbrdfs,
            'name': os.path.basename(img_path)
        }
        result.update(pattern)
        return result
    
    def worker_init_fn(self, worker_id, num_workers=1, rank=1, seed=1, fix_seed=False):
        if fix_seed:
            # Set the worker seed to num_workers * rank + worker_id + seed
            worker_seed = num_workers * rank + worker_id + seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        self.init()