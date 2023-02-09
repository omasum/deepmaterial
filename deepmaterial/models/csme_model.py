from collections import OrderedDict
import logging
from pydoc import render_doc
from turtle import back

from deepmaterial.archs import build_network
from deepmaterial.archs.arch_util import arg_softmax, gumbel_softmax
from deepmaterial.losses import build_loss
from deepmaterial.utils.img_util import color_map, imwrite
import torch.nn.functional as F

from deepmaterial.utils.registry import MODEL_REGISTRY
from deepmaterial.utils.render_util import PlanarSVBRDF, Render, toLDR_torch, torch_norm
from deepmaterial.models.base_model import BaseModel
import torch
import importlib
import os.path as osp
metric_module = importlib.import_module('deepmaterial.metrics')
logger = logging.getLogger('basicsr')


@MODEL_REGISTRY.register()
class CSMEModel(BaseModel):

    def __init__(self, opt):
        super(CSMEModel, self).__init__(opt)
        # define network
        brdf_args = opt['network_g'].pop('brdf_args')
        eval_nbRendering = brdf_args.pop('eval_nbRendering')
        self.renderer = Render(brdf_args)
        brdf_args['nbRendering'] = eval_nbRendering
        brdf_args['toLDR'] = False
        self.renderer_eval = Render(brdf_args)
        self.nc = opt['network_g'].pop('start_nc')
        self.style_n = opt['network_g'].pop('style_n')
        self.net_g = build_network(opt['network_g'])
        if opt.get('network_n', None) is not None:
            self.nGAN = build_network(opt['network_n'])
            self.num_style_feat = opt['network_n']['num_style_feat']
        else:
            self.nGAN = None
        self.net_g = self.model_to_device(self.net_g)
        self.nGAN = self.model_to_device(self.nGAN)
        if not ('print_net_g' in opt and not opt['print_net_g']):
            self.print_network(self.net_g)
            self.print_network(self.nGAN)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True))
            logger.info('Load pre-trained parameters from '+load_path)
        load_path = self.opt['path'].get('pretrain_network_n', None)
        if load_path is not None:
            self.load_network(self.nGAN, load_path, self.opt['path'].get('strict_load_n', True))
            logger.info('Load pre-trained parameters from '+load_path)
            self.init_path = self.opt['path'].get('init_latent', None)
        self.mode = opt['mode'] # n+c, m, n+m, gn+c, gn+m, n+gm

    def set_image(self, img=None, gt=None, l=None, img_name='TestImg', bgr2rgb=False, channel = 'hwc', toHDR=False):
        self.img_name = img_name
        if gt is not None:
            self.gt = PlanarSVBRDF(self.brdf_args).get_svbrdfs(gt)
            #[-1,1]
        if img is not None:
            # img: [n, 3, h, w] or [3, h, w]
            if len(img.shape) == 3:
                self.renderer_eval.nbRendering = 1
            else:
                self.renderer_eval.nbRendering = img.shape[0]
            if l is not None:
                light_pos = torch.from_numpy(l)
                view_pos = light_pos
            else:
                light_pos = self.renderer_eval.lighting.fixedLightsSurface()
                view_pos = self.renderer_eval.camera.fixedView()
            self.light_dir, self.view_dir, self.light_dis, self.pos = self.renderer_eval.torch_generate(view_pos, light_pos)
            if toHDR:
                img = img**2.2
            self.inputs = torch.from_numpy(img).to(self.device)
            if channel == 'hwc':
                self.inputs = self.inputs.permute(2,0,1).contiguous()
            if bgr2rgb:
                self.inputs = self.inputs[::-1, :, :]
        else:
            if self.renderer_eval.nbRendering == 1:
                random_light = False
            else:
                random_light = True
            self.gt[3:4] = torch.ones((1,256,256),dtype=torch.float32)*0.1
            self.gt[4:5] = torch.ones((1,256,256),dtype=torch.float32)*0.5
            self.gt[5:6] = torch.ones((1,256,256),dtype=torch.float32)*0.4
            self.gt[6:7] = torch.ones((1,256,256),dtype=torch.float32)*0.1
            # self.gt[7:8] = torch.ones((1,256,256),dtype=torch.float32)*0.7
            # self.gt[8:9] = torch.ones((1,256,256),dtype=torch.float32)*0.4
            # self.gt[9:10] = torch.ones((1,256,256),dtype=torch.float32)*0.1
            # self.gt[:3] = self.renderer.sphere_normal(padding=True)[0]
            # self.gt[:3] = self.renderer.plane_normal()
            self.inputs, self.light_dir, self.view_dir, self.pos, self.light_dis = self.renderer_eval.render(self.gt, random_light=random_light, colocated=True, toLDR=True)
            self.inputs = self.inputs.to(self.device)

    def optimize_parameters(self, current_iter):
        if current_iter == 1:
            logger.warning('Optimize material parameters and fix network.')
            # self.normal.requires_grad=True
            # self.prob.requires_grad=True
            for p in self.net_g.parameters():
                p.requires_grad = False
            if self.nGAN is not None:
                for p in self.nGAN.parameters():
                    p.requires_grad = False

        self.optimizer_g.zero_grad()
        if 'c' in self.mode:
            # trace, mask = self.assemble_trace()
            # self.output = self.net_g(trace, mask)

            trace, _ = self.assemble_trace_simple()
            material = []
            for i in range(self.nc):
                if trace[i].shape[0] == 0:
                    material.append(torch.zeros(7,device=self.device))
                    continue
                material.append(self.net_g(trace[i], None))
            self.output = torch.stack(material, dim=0)
            # print(torch.abs(self.output-torch.from_numpy([[0.1,0.5,0.4,0.1,0.7,0.4,0.1]])))
            
            self.output_eval = self.output.detach()
        elif 'm' in self.mode:
            self.output = None
            self.output_eval = None

        l_total = 0
        loss_dict = OrderedDict()

        self.rerender, svbrdf = self.eval_render(self.output)
        l_pix = self.cri_pix(self.rerender, self.inputs)
        # l_pix = self.cri_pix(self.get_normal(), self.gt[:3].to(self.device))
        loss_dict['loss_pix'] = l_pix
        l_total += l_pix
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.rerender, self.inputs)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        if hasattr(self, 'gt'):
            l_svbrdf = self.cri_pix(self.gt, svbrdf.cpu())
            loss_dict['loss_svbrdf'] = l_svbrdf
        # self.output.retain_grad()
        l_total.backward()
        self.optimizer_g.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)
        # return l_total.detach().cpu().numpy()
    
    def optimize_material(self):
        total_iter = self.opt['train_material']['total_iter']
        for i in range(total_iter):
            self.optimizer_material.zero_grad()
            rerender, svbrdf = self.eval_render(torch.tanh(self.m_opt), prop_interupt=True)
            loss = self.cri_pix(rerender, self.inputs)
            loss.backward()
            self.optimizer_material.step()
        self.output_eval = torch.tanh(self.m_opt.detach())
    
    
    def assemble_trace(self):
        prob = self.prob
        c,h,w = self.inputs.shape[-3:]
        nc = self.nc
        l = self.light_dir # (1, 3, h, w)
        v = self.view_dir # (1, 3, h, w)
        p = self.pos[:,:2] # (1, 2, h, w)
        d = self.light_dis # (1, 1, h, w)
        n = self.get_normal().unsqueeze(0).detach() # (1, 3, h, w)
        
        label = gumbel_softmax(prob) # (h, w, nc)
        # label = arg_softmax(prob) # (h, w, nc)
        # sorted_input: (c, nc, h,w), while nc is the class dimention in which the measurements has been
        # zerolized by label_rgb.
        # label_shadow = self.inputs.mean(dim=0)>0.01
        # label_plane = self.normal[2,:,:]<0.9
        # label = label * label_shadow.unsqueeze(-1)
        label_rgb = label.unsqueeze(-1).repeat(1,1,1,c).permute(3,2,0,1).contiguous() # (c, nc, h, w)

        ob = (self.inputs.unsqueeze(1) * label_rgb)
        arr = [n,ob,l,v,p,d]
        trace = torch.cat([x.permute(1,0,2,3).repeat(1,nc,1,1) if x is not ob else x for x in arr],dim=0)
        
        # trace: (nc, h*w, c), label: (nc, h*w)
        return trace.permute(1,2,3,0).contiguous().view(nc,h*w,-1), label.permute(2,0,1).view(nc,h*w)

    def assemble_trace_simple(self):
        prob = self.prob
        c,h,w = self.inputs.shape[-3:]
        ln = self.renderer_eval.nbRendering
        nc = self.nc
        l = self.light_dir # (ln, 3, h, w)
        v = self.view_dir.broadcast_to(*l.shape) # (ln, 3, h, w)
        p = self.pos[:,:2].repeat(ln, 1, 1, 1) # (ln, 2, h, w)
        d = self.light_dis # (ln, 1, h, w)
        n = self.get_normal().unsqueeze(0).repeat(ln, 1,1,1) # (ln, 3, h, w)
        ob = self.inputs.unsqueeze(0).view(-1, c, h, w) # (ln, 3, h, w)
        
        label = gumbel_softmax(prob).unsqueeze(0).repeat(ln,1,1,1) # (ln, h, w, nc)
        # label = arg_softmax(prob) # (h, w, nc)
        # label_shadow = self.inputs.mean(dim=0)>0.01
        # label_plane = self.normal[2,:,:]<0.9
        # label = label * label_shadow.unsqueeze(-1)

        arr = [n,ob,l,v,p,d]
        trace = torch.cat([x.permute(0,2,3,1).contiguous() for x in arr],dim=-1) # (ln, h, w, c)
        
        c = trace.shape[-1]
        trace = [(trace*label[:,:,:,i:i+1]).masked_select(label[:,:,:,i:i+1]==1).view(-1, c) for i in range(label.shape[-1])]
        # trace: (nc, ln*h*w, c), label: (nc, ln*h*w)
        return trace, label.permute(3,0,1,2).view(nc,ln*h*w)

    def save_trace(self, path):
        l = self.light_dir/2+0.5 # (1, 3, h, w)
        v = self.view_dir/2+0.5 # (1, 3, h, w)
        p = self.pos/2+0.5 # (1, 3, h, w)
        d = self.light_dis.repeat(1,3,1,1) # (1, 3, h, w)

        d = (d-torch.min(d))/(torch.max(d)-torch.min(d))
        trace = torch.cat([l,v,p,d],dim=2)[0].cpu().permute(1,2,0).contiguous().numpy() # (1,3,h,4*w)

        imwrite(trace, path, float2int=True)

    def save_visual(self, current_iter):
        save_path = osp.join(self.opt['path']['visualization'],f'render_{self.img_name}_{current_iter}.png')
        b_save_path = osp.join(self.opt['path']['visualization'],f'brdf_{self.img_name}_{current_iter}.png')
        i_save_path = osp.join(self.opt['path']['visualization'],f'inputs_{self.img_name}_{current_iter}.png')
        e_save_path = osp.join(self.opt['path']['visualization'],f'error_{self.img_name}_{current_iter}.png')
        c_save_path = osp.join(self.opt['path']['visualization'],f'class_{self.img_name}_{current_iter}.png')
        if hasattr(self, 'gt'):
            gt = self.gt
        else:
            gt = None
        pred = self.reConsMaterial(self.output_eval, True)

        results = self.get_current_visuals(pred.cpu(), gt)

        self.save_visuals(save_path, b_save_path, i_save_path, results)
        imwrite(self.get_error_map().numpy(), e_save_path, float2int=True)
        imwrite(self.get_class_map(), c_save_path, float2int=True)

    def save_visuals(self, path, b_path, i_path, dict, toLDR=True):
        pred = dict['pred']
        gt = dict['gt']
        pbrdf = dict['pbrdf']
        inputs = dict['inputs']
        n,c,h,w = pred.shape
        pred = pred.permute(2,0,3,1).contiguous().view(h,-1,c)
        pbrdf = PlanarSVBRDF.brdf2uint8(pbrdf.unsqueeze(0),False).permute(2,0,3,1).contiguous().view(h,-1,c)
        if toLDR:
            pred = toLDR_torch(pred, sturated=True)
            if gt is not None:
                gt = toLDR_torch(gt, sturated=True)
            inputs = toLDR_torch(inputs, sturated=True)
        if gt is not None:
        # if False:
            gt = gt.permute(2,0,3,1).contiguous().view(h,-1,c)
            imgs = torch.cat([pred, gt], dim=0)
            brdf = dict['brdf']
            brdf = PlanarSVBRDF.brdf2uint8(brdf.unsqueeze(0),False).permute(2,0,3,1).contiguous().view(h,-1,c)
            pbrdf = torch.cat([pbrdf, brdf],dim=0)
        else:
            imgs = pred
        
        imwrite(pbrdf.numpy()[:,:,::-1], b_path, float2int=True)
        imwrite(imgs.numpy()[:,:,::-1], path, float2int=False)
        imwrite(inputs.numpy()[:,:,::-1], i_path, float2int=False)
        
    def get_current_visuals(self, pred, gt = None):
        out_dict = OrderedDict()
        # pred_vis,l,v,p,d = self.renderer.render(pred, random_light=True)
        pred_vis = self.renderer.render(pred, random_light=True, keep_dirs=True)
        if gt is not None:
            gt_vis = self.renderer.render(gt, load_dirs=True)
            out_dict['gt'] = gt_vis
        else:
            out_dict['gt'] = None
        c,h,w = self.inputs.shape[-3:]
        rerender = self.renderer.render(pred.to(self.device), light_dir=self.light_dir,view_dir=self.view_dir,light_dis=self.light_dis,surface=self.pos)
        if self.renderer_eval.nbRendering > 1:
            rerender = rerender.cpu().permute(2,0,3,1).contiguous().view(h,-1,c)
            inputs = self.inputs.cpu().permute(2,0,3,1).contiguous().view(h,-1,c)
            inputs = torch.cat([rerender, inputs], dim=0)
        else:
            if self.renderer.nbRendering > 1:
                rerender.squeeze_(0)
            rerender = rerender.cpu().permute(1,2,0).contiguous()
            inputs = self.inputs.cpu().permute(1,2,0).contiguous()
            inputs = torch.cat([rerender, inputs], dim=0)

        out_dict['pred'] = pred_vis
        out_dict['inputs'] = inputs
        out_dict['pbrdf'] = pred
        if hasattr(self, 'gt'):
            out_dict['brdf'] = self.gt
        return out_dict
    
    def initialization(self):
        self.init_opt()
        self.init_training_settings()

    def load_latent(self, path):
        latent = torch.load(path)
        z, noise = latent[-1], latent[:-1]
        return (z, noise)

    def make_noise(self, batch, num_noise):
        if num_noise == 1:
            noises = torch.randn(batch, self.num_style_feat, device=self.device)
        else:
            noises = torch.randn(batch, num_noise, self.num_style_feat, device=self.device)
        return noises
    
    def init_opt(self):
        self.net_g.eval()
        if self.nGAN is not None:
            self.nGAN.eval()
        h,w = self.inputs.shape[-2:]

        self.light_dir, self.view_dir, self.light_dis, self.pos = \
            self.light_dir.to(self.device), self.view_dir.to(self.device), self.light_dis.to(self.device), self.pos.to(self.device)
        self.prob = torch_norm(torch.rand((h, w, self.nc),dtype=torch.float32).to(self.device),dim=-1)
        if 'm' in self.mode and 'gm' not in self.mode:
            self.material = torch.ones((7,h,w),dtype=torch.float32).to(self.device)*0.2
        elif 'gm' in self.mode:
            self.material = self.gt[3:].to(self.device)
        if ('n' in self.mode and 'gn' not in self.mode) or ('gn' in self.mode and not hasattr(self, 'gt')):
            self.normal = self.renderer_eval.plane_normal().to(self.device)
            if self.style_n and 'gn' not in self.mode:
                if self.init_path is not None:
                    self.normal = self.load_latent(self.init_path)
                else:
                    z = self.make_noise(1, 14)
                    noise = [getattr(self.nGAN.noises, f'noise{i}') for i in range(self.nGAN.num_layers)]
                    self.normal = (z, noise)
        elif 'gn' in self.mode and hasattr(self, 'gt'):
            self.normal = self.gt[:3].to(self.device)
        if 'c' in self.mode:
            self.prob.requires_grad = True
        elif 'm' in self.mode and 'gm' not in self.mode:
            self.material.requires_grad=True
        if 'n' in self.mode and 'gn' not in self.mode:
            if self.style_n:
                self.normal[0].requires_grad=True
                for noise in self.normal[1]:
                    noise.requires_grad=True
            else:
                self.normal.requires_grad = True

        self.m_opt = torch.zeros((self.nc, 7), dtype=torch.float32, device=self.device)
        self.m_opt.requires_grad=True
    
    def get_normal(self):
        if self.style_n:
            pos = torch_norm(self.nGAN(self.normal[0], noise=self.normal[1], input_is_latent=True)[0][0], dim=0)
        else:
            c, h, w = self.normal.shape
            pos = torch_norm(self.normal,dim=0)
        return pos

    def add_prob(self, error):
        self.nc += 1
        if self.nc == 2:
            self.prob.requires_grad=True
        threshod = torch.max(error)*0.5
        mask = (error>threshod).to(self.device)
        max_prob = torch.max(self.prob.detach() * mask,dim=-1,keepdim=True)[0]+0.1
        self.prob = torch_norm(torch.cat([self.prob.detach(),max_prob],dim=-1),dim=-1)
        self.prob.requires_grad=True

    def reConsMaterial(self, pred, vis=False):
        # pred is the predicted material parameters (nc, 7)
        if vis:
            category = F.one_hot(torch.argmax(self.prob.detach(),dim=-1), num_classes=self.nc) 
            n = self.get_normal().detach()
            if 'm' in self.mode:
                material = self.material.detach()
        else:
            category = gumbel_softmax(self.prob)
            # category = arg_softmax(self.prob)
            n = self.get_normal().detach()
            if 'm' in self.mode:
                material = self.material
        h,w = self.inputs.shape[-2:]
        if 'c' in self.mode:
            material = torch.matmul(category.float(), pred).permute(2,0,1)
            # material = pred.view(1,1,self.nc,-1).repeat(h,w,1,1)*category.unsqueeze(-1)
            # material = torch.sum(material, dim=-2).permute(2,0,1)# 7, h, w
        svbrdf = torch.cat([n,material],dim=0)
        # svbrdf = self.gt.to(self.device)
        return svbrdf

    def eval_render(self, pred, prop_interupt=False):
        svbrdf = self.reConsMaterial(pred, vis=prop_interupt)
        rerender = self.renderer_eval.render(svbrdf, colocated=True, light_dir=self.light_dir,view_dir=self.view_dir,light_dis=self.light_dis,surface=self.pos)
        return rerender, svbrdf

    def init_training_settings(self):
        train_opt = self.opt['train']

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        if self.mode=='c' or self.mode=='gn+c':
            optim_params = [self.prob]
        elif self.mode=='n' or self.mode=='n+gm':
            if self.style_n:
                optim_params=self.normal[1]+[self.normal[0]]
            else:
                optim_params = [self.normal]
        elif self.mode=='m' or self.mode=='gn+m':
            optim_params = [self.material]
        elif self.mode=='n+c':
            if self.style_n:
                optim_params=self.normal[1]+[self.normal[0], self.prob]
            else:
                optim_params = [self.normal, self.prob]
        elif self.mode=='n+m':
            if self.style_n:
                optim_params=self.normal[1]+[self.normal[0], self.material]
            else:
                optim_params = [self.normal, self.material]
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        
        
        train_opt = self.opt['train_material']
        optim_type = train_opt['optim'].pop('type')
        self.optimizer_material = self.get_optimizer(optim_type, [self.m_opt], **train_opt['optim'])
        self.optimizers.append(self.optimizer_material)

    def feed_data(self, data):
        return

    def get_error_map(self, eps = 1e-12):
        pred = torch.clip(self.rerender.detach().cpu(),0.0, 1.0)
        gt = torch.clip(self.inputs.cpu(),0.0, 1.0)
        error = F.l1_loss(pred, gt, reduction='none')
        if len(error.shape) == 4:
            h,w = error.shape[-2:]
            error = error.view(-1, h,w)
        error = torch.mean(error,dim=0,keepdim=True)
        return error.permute(1,2,0).contiguous()

    def get_class_map(self):
        tag = torch.argmax(self.prob.detach(),dim=-1)
        tag = tag/(self.nc-1)
        tag = color_map(tag.cpu().numpy())
        return tag

    # def save_para(self, current_iteration):
