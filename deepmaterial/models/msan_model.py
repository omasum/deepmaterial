from collections import OrderedDict
from copy import deepcopy
from email.mime import base
import logging
from deepmaterial.archs import build_network
from deepmaterial.models.radn_model import RADNModel
from deepmaterial.utils.img_util import imwrite,tensor2img
from deepmaterial.utils.registry import MODEL_REGISTRY
from deepmaterial.utils.render_util import PolyRender, RectLighting, torch_norm
import torch, os
import importlib
import numpy as np
import torch.nn.functional as F
import torchvision
metric_module = importlib.import_module('deepmaterial.metrics')
logger = logging.getLogger('deepmaterial')


@MODEL_REGISTRY.register()
class MSANModel(RADNModel):

    def __init__(self, opt):
        self.optPattern = opt['network_g'].get('optPattern', False)
        super(MSANModel, self).__init__(opt)
        self.optPattern = self.opt['network_g'].get('optPattern', False)
        if self.optPattern:
            self.initPattern()

    def initRender(self):
        super().initRender()
        tex = self.brdf_args.get('texture', None)
        self.rendererPoint = self.renderer
        self.renderer = PolyRender(self.brdf_args, device=self.device)
        if isinstance(tex, list):
            self.renderer.nbRendering = len(tex) + self.renderer.nbRendering
            self.lighting = []
            config = deepcopy(self.brdf_args)
            for t in tex:
                config['texture'] = t
                self.lighting.append(RectLighting(config, device=self.device))
        else:
            self.renderer.nbRendering = 1 + self.renderer.nbRendering
            self.lighting = RectLighting(self.brdf_args, device=self.device)
            if tex is None:
                self.lighting.initGaussianConv()
            self.renderer.lighting = self.lighting
            viewPos = self.renderer.camera.fixedView()
            self.lDir, self.vDir, __, __ = self.renderer.torch_generate(viewPos, self.lighting.vertices, normLight=False)
            self.lDir, self.vDir = self.lDir.to(self.device), self.vDir.to(self.device)
    
    def initPattern(self):
        self.pattern = torch.zeros(*self.opt.get('patternSize', (1, 3, 1024, 1024)), dtype=torch.float32, device=self.device)
        self.pattern.requires_grad = True
        params = {'params':[self.pattern]}
        params.update(**self.opt['train']['optim_pattern'])
        self.optimizer_g.add_param_group(params)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        if current_iter==1:
            logger.warning('Train all the parameters.')
            for p in self.net_g.parameters():
                p.requires_grad = True
        if self.opt.get('network_d'):
            # optimize net_g
            for p in self.net_d.parameters():
                p.requires_grad = False
        
        self.optimizer_g.zero_grad()
        if self.optPattern:
            self.inputs = self.render_input(torch.tanh(self.pattern))
            output = self.net_g(self.inputs, self.pattern)
        else:
            output = self.net_g(self.inputs)
        l_total = self.computeLoss(output, loss_dict)
        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            if self.optPattern:
                self.inputs = self.render_input(torch.tanh(self.pattern))
                output = self.net_g(self.inputs, self.pattern)
            else:
                output = self.net_g(self.inputs)
            normal, diffuse,roughness,specular = torch.split(output,[3,3,1,3],dim=1)
            normal = torch_norm(normal,dim=1)
            self.output = torch.cat([normal, diffuse, roughness, specular],dim=1)
        self.net_g.train()

    def render_input(self, pattern):
        textureMode = 'Torch'
        pattern = pattern / 2 + 0.5
        self.lighting.initTexture(pattern, textureMode=textureMode)
        inputs = self.renderer.render(self.svbrdf, light_dir=self.lDir, view_dir=self.vDir, n_xy=False, toLDR=False)
        return inputs*2-1

    def eval_render(self, pred, gt):
        self.renderer.lutToDevice(pred.device)
        if self.optPattern:
            textureMode = 'Torch'
            pattern = torch.tanh(self.pattern.detach())/2+0.5
            self.lighting.initTexture(pattern, textureMode=textureMode)
        if isinstance(self.lighting, list):
            rerender = []
            gtrender = []
            for lighting in self.lighting:
                lighting.lodToDevice(pred.device)
                self.renderer.lighting = lighting
                rerender.append(self.renderer.render(pred, n_xy=False))
                gtrender.append(self.renderer.render(gt, n_xy=False))
                lighting.lodToDevice(self.device)
            rerender = torch.stack(rerender, dim=1)
            gtrender = torch.stack(gtrender, dim=1)
        else:
            self.lighting.lodToDevice(pred.device)
            rerender = self.renderer.render(pred, n_xy=False).unsqueeze(1)
            gtrender = self.renderer.render(gt, n_xy=False).unsqueeze(1)
            self.lighting.lodToDevice(self.device)
            
        self.renderer.lutToDevice(self.device)

        rerenderPoint = self.rendererPoint.render(pred, n_xy=False, keep_dirs=True)
        gtrenderPoint = self.rendererPoint.render(gt, n_xy=False, load_dirs=True)
        rerender = torch.cat([rerenderPoint,rerender], dim=1)
        gtrender = torch.cat([gtrenderPoint, gtrender], dim=1)
        return rerender, gtrender

    def computeLoss(self, output, loss_dict, isDesc=False):
        normal, diffuse,roughness,specular = torch.split(output,[3,3,1,3],dim=1)
        normal = torch_norm(normal,dim=1)
        return self.brdfLoss(normal, diffuse, roughness, specular, loss_dict, isDesc)

    def eval_render_train_area(self, pred, gt):
        self.renderer.lighting.set_lamp_intensity(3)
        gt_render = self.renderer.render(gt, light_dir=self.lDir, view_dir=self.vDir, n_xy=False, toLDR=False)
        pred_render = self.renderer.render(pred, light_dir=self.lDir, view_dir=self.vDir, n_xy=False, toLDR=False)
        self.renderer.lighting.reset_lamp_intensity()
        return gt_render, pred_render
    
    def save(self, epoch, current_iter):
        super().save(epoch, current_iter)
        if self.optPattern:
            save_filename = f'pattern_{current_iter}.pth'
            save_path = os.path.join(self.opt['path']['models'], save_filename)
            torch.save(self.pattern, save_path)
    
    def load_network(self, net, load_path, strict=True, param_key='params'):
        super().load_network(net, load_path, strict, param_key)
        if self.optPattern:
            load_dir, base = os.path.dirname(load_path)
            current_iter = base.split('_')[1].split('.')[0]
            self.pattern = torch.load(os.path.join(load_dir, f'pattern_{current_iter}.pth'))
            self.pattern.requires_grad = True
    
@MODEL_REGISTRY.register()
class MSANTwoStreamModel(MSANModel):
    def __init__(self, opt):
        super(MSANModel, self).__init__(opt)
        self.optPattern = self.opt['network_g'].get('optPattern', False)
        self.renderer.nbRendering += 1
    
    def initNetworks(self, printNet=True):
        super().initNetworks(printNet)
        netPattern = self.opt.get('network_pattern', None)
        self.net_pattern = self.buildNet(netPattern, 'pattern')

    def setup_optimizers(self):
        super().setup_optimizers()
        train_opt = self.opt['train']['optim_pattern']
        optim_params = [param for param in self.net_pattern.parameters() if param.requires_grad]
        optim_type = train_opt.pop('type')
        self.optimizer_pattern = self.get_optimizer(optim_type, optim_params, **train_opt)
        self.optimizers.append(self.optimizer_pattern)
    
    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        if current_iter==1:
            logger.warning('Train all the parameters.')
            for p in self.net_g.parameters():
                p.requires_grad = True
        if self.opt.get('network_d'):
            # optimize net_g
            for p in self.net_d.parameters():
                p.requires_grad = False
        
        self.optimizer_g.zero_grad()
        self.optimizer_pattern.zero_grad()

        self.secPattern = self.net_pattern(self.inputs)
        self.secInputs = self.render_input(self.secPattern)

        output = self.net_g([self.inputs,self.secInputs], [self.pattern, self.secPattern])

        l_total = self.computeLoss(output, loss_dict)
        l_total.backward()

        self.optimizer_pattern.step()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def feed_data(self, data):
        super().feed_data(data)
        self.pattern = data['pattern'].cuda()
    
    def test(self):
        self.net_g.eval()
        self.net_pattern.eval()
        with torch.no_grad():
            self.secPattern = self.net_pattern(self.inputs)
            self.secInputs = self.render_input(self.secPattern)
            output = self.net_g([self.inputs, self.secInputs], [self.pattern, self.secPattern])
            normal, diffuse,roughness,specular = torch.split(output,[3,3,1,3],dim=1)
            normal = torch_norm(normal,dim=1)
            self.output = torch.cat([normal, diffuse, roughness, specular],dim=1)
        self.net_pattern.train()
        self.net_g.train()

    def eval_render(self, pred, gt):
        self.renderer.lutToDevice(pred.device)
        textureMode = 'Torch'
        gtrender = []
        rerender = []
        for pattern in [self.secPattern, self.pattern]:
            pattern = pattern.detach()/2+0.5
            self.lighting.initTexture(pattern, textureMode=textureMode)
            self.lighting.lodToDevice(pred.device)
            rerender.append(self.renderer.render(pred, n_xy=False).unsqueeze(1))
            gtrender.append(self.renderer.render(gt, n_xy=False).unsqueeze(1))
            self.lighting.lodToDevice(self.device)
        
        self.renderer.lutToDevice(self.device)

        rerender.append(self.rendererPoint.render(pred, n_xy=False, keep_dirs=True))
        gtrender.append(self.rendererPoint.render(gt, n_xy=False, load_dirs=True))
        rerender.reverse()
        gtrender.reverse()

        rerender = torch.cat(rerender, dim=1)
        gtrender = torch.cat(gtrender, dim=1)
        return rerender, gtrender
    
    def save_visuals(self, path, pred, gt):
        super().save_visuals(path, pred, gt)
        pattern = torch.cat([self.pattern, self.secPattern], dim=3)
        outputImg = tensor2img(pattern/2+0.5)
        imwrite(outputImg, path.replace('svbrdf', 'pattern'), float2int=False)        
            
    def save(self, epoch, current_iter):
        super(MSANModel, self).save(epoch, current_iter)
        self.save_network(self.net_pattern, 'net_pattern', current_iter)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        super(MSANModel, self).load_network(net, load_path, strict, param_key)
