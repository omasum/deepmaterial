#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
from ..vector_util import numpy_norm, torch_norm, torch_dot
from ..img_util import img2tensor, toLDR_torch, toHDR_torch, preprocess
from abc import abstractmethod

lightDistance = 2.14
viewDistance = 2.75 # 39.98 degrees FOV

class BRDF:
    def __init__(self, opt):
        self.opt = opt

    def GGX(self, NoH, roughness, eps = 1e-12):
        alpha = roughness  * roughness
        eps = torch.ones_like(roughness)*eps
        tmp = alpha / torch.max(eps,  (NoH * NoH * (alpha * alpha - 1.0) + 1.0  ) )
        return tmp * tmp * (1/torch.pi)

    def SmithG(self, NoV, NoL, roughness, eps = 1e-12):
        def _G1(NoM, k):
            return NoM / (NoM * (1.0 - k ) + k)

        eps = torch.ones_like(roughness)*eps
        k = torch.max(eps, roughness * roughness * 0.5)
        return _G1(NoL,k) * _G1(NoV, k)

    def Fresnel(self, F0, VoH, method='schlick'):
        if method == 'schlick':
            coeff = (1-VoH) ** 5
        else:
            coeff = 2**(VoH * (-5.55473 * VoH - 6.98316))
        return F0 + (1.0 - F0) * coeff
    
    @abstractmethod
    def eval(self):
        pass    

    @abstractmethod    
    def sample(self):
        pass
    
    def norm(self):
        pass

class PlanarSVBRDF(BRDF):
    def __init__(self, opt):
        super().__init__(opt)
        self.size = opt['size']
    def split_img(self, imgs, split_num=5, split_axis=1, concat=True, svbrdf_norm=True):
        """Split input image to $split_num images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        split_num (int): number of input images containing
        split_axis (int): which axis the split images is concated.

    Returns:
        list[ndarray]: Split images.
    """
        order = self.opt['order']
        gamma_correct = self.opt.get('gamma_correct', '')

        def _norm(img, gamma_correct='pdrs', order='pndrs'):
            n,d,r,s = None, None, None, None
            if 'ndrs' in order:
                n,d,r,s = img[-4:]
            elif order == 'dnrs':
                d,n,r,s = img
            r = np.mean(r,axis=-1,keepdims=True)
            if 'd' in gamma_correct:
                d = d**2.2
            if 'r' in gamma_correct:
                r = r**2.2
            if 's' in gamma_correct:
                s = s**2.2
            result = []
            if order == 'pndrs' or order=='ndrs':
                result = [n,d,r,s]
            elif order == 'dnrs':
                result = [d,n,r,s]
            result = [preprocess(x) for x in result]
            return result

        if split_num == 1:
            return imgs
        else:
            if isinstance(imgs, list):
                imglist = [np.split(v,split_num,axis=split_axis) for v in imgs]
                if svbrdf_norm:
                    imglist = [_norm(v,gamma_correct,order) for v in imglist]
                if concat:
                    return [np.concatenate(v,axis=-1) for v in imglist]
            else:
                imglist = np.split(imgs,split_num,axis=split_axis)
                if svbrdf_norm:
                    imglist = _norm(imglist, gamma_correct, order)
                if concat:
                    return np.concatenate(imglist,axis=-1)

    def brdf2uint8(self, svbrdf, n_xy=True, r_single=True, gamma_correct=None):
        if gamma_correct is None:
            gamma_correct = self.opt.get('gamma_correct', '')
        if not n_xy:
            svbrdf = svbrdf/2+0.5
            n = svbrdf[:,0:3]
            d = svbrdf[:,3:6]
            if r_single:
                r = svbrdf[:,6:7]
                s = svbrdf[:,7:10]
                r = torch.cat([r]*3, dim=1)
            else:
                r = svbrdf[:,6:9]
                s = svbrdf[:,9:12]
        else:
            n = svbrdf[:, 0:2]
            n = self.unsqueeze_normal(n)/2+0.5
            d = svbrdf[:,2:5]/2+0.5
            if r_single:
                r = svbrdf[:,5:6]/2+0.5
                s = svbrdf[:,6:9]/2+0.5
                r = torch.cat([r]*3, dim=1)                
            else:
                r = svbrdf[:,5:8]/2+0.5
                s = svbrdf[:,8:11]/2+0.5
        if 'd' in gamma_correct:
            d = d**0.4545
        if 'r' in gamma_correct:
            r = r**0.4545
        if 's' in gamma_correct:
            s = s**0.4545
        
        result = torch.cat([n,d,r,s], dim=3)
        return result
    
    def get_svbrdfs(self, imgs, normalization=False, num=5, axis=1, concat=True, norm=True):
        return img2tensor(self.split_img(imgs, split_num=num, split_axis=axis, concat=concat, svbrdf_norm=norm)\
            ,bgr2rgb=False, float32=True, normalization=normalization)
    
    def homo2sv(self, brdf):
        if len(brdf.shape) >= 2:
            n, c = brdf.shape
            svbrdf = torch.ones((n,c,self.size, self.size),dtype=torch.float32)*(brdf.unsqueeze(-1).unsqueeze(-1))
        else:
            c = brdf.shape[0]
            svbrdf = torch.ones((c, self.size, self.size), dtype=torch.float32)*(brdf.unsqueeze(1).unsqueeze(1))
        return svbrdf
       
    def _seperate_brdf(self, svbrdf, n_xy=False, r_single=True):
        if svbrdf.dim() == 4:
            b,c,h,w = svbrdf.shape
        else:
            b = 0
            svbrdf = svbrdf.unsqueeze(0)
        
        if self.opt['order'] == 'pndrs' or self.opt['order'] == 'ndrs':
            if not n_xy:
                n = svbrdf[:,0:3]
                d = svbrdf[:,3:6]
                if r_single:
                    r = svbrdf[:,6:7]
                    s = svbrdf[:,7:10]
                else:
                    r = svbrdf[:, 6:7]
                    s = svbrdf[:, 9:12]
            else:
                n = svbrdf[:, 0:2]
                n = self.unsqueeze_normal(n)
                d = svbrdf[:,2:5]
                if r_single:
                    r = svbrdf[:,5:6]
                    s = svbrdf[:,6:9]
                else:
                    r = svbrdf[:,5:6]
                    s = svbrdf[:,8:11]
        elif self.opt['order'] == 'dnrs':
            d = svbrdf[:,0:3]
            n = svbrdf[:,3:6]
            r = svbrdf[:,6:7]
            s = svbrdf[:,7:10]
        if b != 0:
            n = n.unsqueeze(1)
            d = d.unsqueeze(1)
            r = r.unsqueeze(1)
            s = s.unsqueeze(1)
        return n,d,r,s

    def squeeze_brdf(self, svbrdf, n_xy=True):
        n,d,r,s = self._seperate_brdf(svbrdf)
        if n_xy:
            n = self.squeeze_normal(n)
        n = n.squeeze(0)
        d = d.squeeze(0)
        r = r.squeeze(0)
        s = s.squeeze(0)
        return torch.cat([n,d,r,s],dim=0)

    def unsqueeze_brdf(self, svbrdf, n_xy = True, r_single = True):
        n,d,r,s = self._seperate_brdf(svbrdf, n_xy=n_xy, r_single=r_single)
        if n_xy:
            n = self.unsqueeze_normal(n)
        n = n.squeeze(0)
        d = d.squeeze(0)
        r = r.squeeze(0)
        s = s.squeeze(0)
        if r_single:
            svbrdf = torch.cat([n,d,r,r,r,s], dim=0)
        else:
            svbrdf = torch.cat([n,d,r,s], dim=0)
        return svbrdf

    def squeeze_normal(self, n):
        '''
            n: normal, shape: (3, h, w)
            return: svbrdf, shape:( 2,h,w)
        '''
        n = torch_norm(n, 1)[:,:2]
        return n
    
    def unsqueeze_normal(self, n):
        b,c,h,w = n.shape
        z = torch.ones((b, 1, h, w),dtype=torch.float32) - (n**2).sum(dim=1)
        n = torch.cat([n, z], dim=1)
        n = torch_norm(n, 1)
        return n

    def eval(self, n,d,r,s,l,v,dis, use_spec=True, per_point=False, intensity=None, useFre=True, eps=1e-12):
        trEps = torch.ones_like(dis)*eps
        
        r = r*0.5+0.5
        # r = torch.ones_like(r)*0.6
        d = d*0.5+0.5
        # d = torch.ones_like(d)*0.6
        s = s*0.5+0.5
        # s = torch.ones_like(s)*0.6
        if per_point:
            dim = -2
        else:
            dim = -3
        n = torch_norm(n, dim=dim)
        h = torch_norm((l+v) * 0.5 , dim=dim)

        NoH = torch_dot(n,h, dim=dim)
        NoV = torch_dot(n,v, dim=dim)
        NoL = torch_dot(n,l, dim=dim)
        VoH = torch_dot(v,h, dim=dim)

        NoH = torch.max(NoH, trEps)
        NoV = torch.max(NoV, trEps)
        NoL = torch.max(NoL, trEps)
        VoH = torch.max(VoH, trEps)

        f_d = d * (1/torch.pi)

        D = self.GGX(NoH,r)
        G = self.SmithG(NoV, NoL, r)
        F = self.Fresnel(s, VoH)
        f_s = D * G * F / (4.0 * NoL * NoV + eps)

        if use_spec:
            res =  (f_d + f_s) * NoL
        else:
            res = (f_d) * NoL
        if intensity is not None:
            res *= intensity * math.pi /dis
        return res

class Render():
    def __init__(self, opt):
        self.opt = opt
        self.size = opt['size']
        self.nbRendering = opt['nbRendering']
        self.useAugmentation = opt.get('useAug',False)
        if opt.get('lampIntensity', False):
            self.lampIntensity = opt['lampIntensity']

    # ---------------------- BRDF ---------------------------------    
    def split_img(self, imgs, split_num=5, split_axis=1, concat=True, svbrdf_norm=True):
        """Split input image to $split_num images.

        Args:
            imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
            split_num (int): number of input images containing
            split_axis (int): which axis the split images is concated.

        Returns:
            list[ndarray]: Split images.
        """
        order = self.opt['order']
        gamma_correct = self.opt.get('gamma_correct', '')

        def _norm(img, gamma_correct='pdrs', order='pndrs'):
            n,d,r,s = None, None, None, None
            if 'ndrs' in order:
                n,d,r,s = img[-4:]
            elif order == 'dnrs':
                d,n,r,s = img
            r = np.mean(r,axis=-1,keepdims=True)
            if 'd' in gamma_correct:
                d = d**2.2
            if 'r' in gamma_correct:
                r = r**2.2
            if 's' in gamma_correct:
                s = s**2.2
            result = []
            if order == 'pndrs' or order=='ndrs':
                result = [n,d,r,s]
            elif order == 'dnrs':
                result = [d,n,r,s]
            result = [preprocess(x) for x in result]
            return result

        if split_num == 1:
            return imgs
        else:
            if isinstance(imgs, list):
                imglist = [np.split(v,split_num,axis=split_axis) for v in imgs]
                if svbrdf_norm:
                    imglist = [_norm(v,gamma_correct,order) for v in imglist]
                if concat:
                    return [np.concatenate(v,axis=-1) for v in imglist]
            else:
                imglist = np.split(imgs,split_num,axis=split_axis)
                if svbrdf_norm:
                    imglist = _norm(imglist, gamma_correct, order)
                if concat:
                    return np.concatenate(imglist,axis=-1)

    def brdf2uint8(self, svbrdf, n_xy=True, r_single=True, gamma_correct=None):
        if gamma_correct is None:
            gamma_correct = self.opt.get('gamma_correct', '')
        if not n_xy:
            svbrdf = svbrdf/2+0.5
            n = svbrdf[:,0:3]
            d = svbrdf[:,3:6]
            if r_single:
                r = svbrdf[:,6:7]
                s = svbrdf[:,7:10]
                r = torch.cat([r]*3, dim=1)
            else:
                r = svbrdf[:,6:9]
                s = svbrdf[:,9:12]
        else:
            n = svbrdf[:, 0:2]
            n = self.unsqueeze_normal(n)/2+0.5
            d = svbrdf[:,2:5]/2+0.5
            if r_single:
                r = svbrdf[:,5:6]/2+0.5
                s = svbrdf[:,6:9]/2+0.5
                r = torch.cat([r]*3, dim=1)                
            else:
                r = svbrdf[:,5:8]/2+0.5
                s = svbrdf[:,8:11]/2+0.5
        if 'd' in gamma_correct:
            d = d**0.4545
        if 'r' in gamma_correct:
            r = r**0.4545
        if 's' in gamma_correct:
            s = s**0.4545
        
        result = torch.cat([n,d,r,s], dim=3)
        return result
    
    def get_svbrdfs(self, imgs, normalization=False, num=5, axis=1, concat=True, norm=True):
        return img2tensor(self.split_img(imgs, split_num=num, split_axis=axis, concat=concat, svbrdf_norm=norm)\
            ,bgr2rgb=False, float32=True, normalization=normalization)
    
    def homo2sv(self, brdf):
        if len(brdf.shape) >= 2:
            n, c = brdf.shape
            svbrdf = torch.ones((n,c,self.size, self.size),dtype=torch.float32)*(brdf.unsqueeze(-1).unsqueeze(-1))
        else:
            c = brdf.shape[0]
            svbrdf = torch.ones((c, self.size, self.size), dtype=torch.float32)*(brdf.unsqueeze(1).unsqueeze(1))
        return svbrdf
       
    def _seperate_brdf(self, svbrdf, n_xy=False, r_single=True):
        if svbrdf.dim() == 4:
            b,c,h,w = svbrdf.shape
        else:
            b = 0
            svbrdf = svbrdf.unsqueeze(0)
        
        if self.opt['order'] == 'pndrs' or self.opt['order'] == 'ndrs':
            if not n_xy:
                n = svbrdf[:,0:3]
                d = svbrdf[:,3:6]
                if r_single:
                    r = svbrdf[:,6:7]
                    s = svbrdf[:,7:10]
                else:
                    r = svbrdf[:, 6:7]
                    s = svbrdf[:, 9:12]
            else:
                n = svbrdf[:, 0:2]
                n = self.unsqueeze_normal(n)
                d = svbrdf[:,2:5]
                if r_single:
                    r = svbrdf[:,5:6]
                    s = svbrdf[:,6:9]
                else:
                    r = svbrdf[:,5:6]
                    s = svbrdf[:,8:11]
        elif self.opt['order'] == 'dnrs':
            d = svbrdf[:,0:3]
            n = svbrdf[:,3:6]
            r = svbrdf[:,6:7]
            s = svbrdf[:,7:10]
        if b != 0:
            n = n.unsqueeze(1)
            d = d.unsqueeze(1)
            r = r.unsqueeze(1)
            s = s.unsqueeze(1)
        return n,d,r,s

    def squeeze_brdf(self, svbrdf, n_xy=True):
        n,d,r,s = self._seperate_brdf(svbrdf)
        if n_xy:
            n = self.squeeze_normal(n)
        n = n.squeeze(0)
        d = d.squeeze(0)
        r = r.squeeze(0)
        s = s.squeeze(0)
        return torch.cat([n,d,r,s],dim=0)

    def unsqueeze_brdf(self, svbrdf, n_xy = True, r_single = True):
        n,d,r,s = self._seperate_brdf(svbrdf, n_xy=n_xy, r_single=r_single)
        if n_xy:
            n = self.unsqueeze_normal(n)
        n = n.squeeze(0)
        d = d.squeeze(0)
        r = r.squeeze(0)
        s = s.squeeze(0)
        if r_single:
            svbrdf = torch.cat([n,d,r,r,r,s], dim=0)
        else:
            svbrdf = torch.cat([n,d,r,s], dim=0)
        return svbrdf

    def squeeze_normal(self, n):
        '''
            n: normal, shape: (3, h, w)
            return: svbrdf, shape:( 2,h,w)
        '''
        n = torch_norm(n, 1)[:,:2]
        return n
    
    def unsqueeze_normal(self, n):
        b,c,h,w = n.shape
        z = torch.ones((b, 1, h, w),dtype=torch.float32) - (n**2).sum(dim=1)
        n = torch.cat([n, z], dim=1)
        n = torch_norm(n, 1)
        return n
    
    # ---------------------- Render -------------------------------
    EPS = 1e-12
    def render(self, svbrdf, single_pos=False, obj_pos=None, light_pos=None, view_pos=None, keep_dirs=False, load_dirs=False, light_dir=None,\
         view_dir=None, light_dis=None, surface=None, random_light=True, colocated=False,\
              n_xy=False, r_single=True, toLDR=None, use_spec = True, per_point=False, isAmbient=False):
        assert not (light_dir is not None and light_pos is not None), ('Given two type of lighting initilization, please choose one type (light_dir, light_pos)')
        assert not (view_dir is not None and view_pos is not None), ('Given two type of view initilization, please choose one type (view_dir, view_pos)')
        if light_dir is not None or light_pos is not None:
            random_light = False
        if toLDR is None:
            toLDR = self.opt['toLDR']
        if light_dir is None:
            if random_light:
                light_pos = self.fullRandomLightsSurface()
            elif light_pos is not None:
                light_pos = torch.from_numpy(np.array(light_pos)).view(1,3)
            else:
                light_pos = self.fixedLightsSurface()
            if colocated:
                view_pos = light_pos
            elif view_pos is not None:
                view_pos = torch.from_numpy(np.array(view_pos)).view(1,3)
            else:
                view_pos = self.fixedView()

            if obj_pos is None and single_pos:
                obj_pos = torch.rand((self.nbRendering, 3), dtype=torch.float32)*2-1
                obj_pos[:, 2] = 0.0
            light_dir, view_dir, light_dis, surface = self.torch_generate(view_pos, light_pos, pos=obj_pos)
        if keep_dirs:
            self.light_dir = light_dir
            self.view_dir = view_dir
            self.light_dis = light_dis
            self.surface = surface
        if load_dirs:
            light_dir, view_dir, light_dis, surface = self.light_dir, self.view_dir, self.light_dis, self.surface
        n,d,r,s = self._seperate_brdf(svbrdf, n_xy, r_single)
        render_result = self._render(n,d,r,s, light_dir, view_dir, light_dis, use_spec=use_spec, per_point=per_point, isAmbient=isAmbient)
        if toLDR:
            render_result = toLDR_torch(render_result)
            render_result = toHDR_torch(render_result)
        return render_result, light_dir, view_dir, surface, light_dis

    def render_panel_single_point(self, brdf, l=None, wi=None, dis=None, wo=None, n_xy=False,\
         r_single=True, toLDR=None):
        # brdf (9/10, 1)
        # l is a panel lighting matrix, in which the element is the intensity of lighting panel.
        # l: (1,hl,wl)
        # wi is a panel lighting matrix, in which the element is direction from the sample point in lighting panel to surface point.
        # if wi is given, the distance need to be given too.
        # wi: (3,hl,wl)
        # wo: (3,ho,ho)
        # dis: (1,hl,wl)
        if toLDR is None:
            toLDR = self.opt['toLDR']
        h, w = wo.shape[-2:]
        wi = wi.permute(1,0).contiguous().view(1,-1,3,1,1).repeat(h*w, 1, 1,1,1)
        l = l.permute(1,0).contiguous().view(1,-1,3,1,1).repeat(h*w, 1, 1,1,1)
        wo = wo.permute(1,2,0).contiguous().view(-1, 1, 3, 1,1).repeat(1,l.shape[1],1,1,1)
        if dis is not None:
            dis = dis.permute(1,2,0).contiguous().view(1,-1, 1, 1, 1).repeat(h*w, 1, 1,1,1)
            l = l/dis**2
        else:
            dis = torch.ones_like(wi)
        v = wo
        n,d,r,s = self._seperate_brdf(brdf, n_xy, r_single)
        n.unsqueeze_(0)
        d.unsqueeze_(0)
        r.unsqueeze_(0)
        s.unsqueeze_(0)
        render_result = self._render(n,d,r,s, wi, v, dis, per_point=False, intensity=l)
        if toLDR:
            render_result = toLDR_torch(render_result)
            render_result = toHDR_torch(render_result)
        render_result = render_result.sum(1).view(h,w,3).permute(2,0,1).contiguous()
        return render_result

    INV_PI = 1.0 / math.pi
    def GGX(self, NoH, roughness):
        alpha = roughness  * roughness
        eps = torch.ones_like(roughness)*self.EPS
        tmp = alpha / torch.max(eps,  (NoH * NoH * (alpha * alpha - 1.0) + 1.0  ) )
        return tmp * tmp * self.INV_PI

    def SmithG(self, NoV, NoL, roughness):
        def _G1(NoM, k):
            return NoM / (NoM * (1.0 - k ) + k)

        eps = torch.ones_like(roughness)*self.EPS
        k = torch.max(eps, roughness * roughness * 0.5)
        return _G1(NoL,k) * _G1(NoV, k)

    def Fresnel(self, F0, VoH):
        coeff = VoH * (-5.55473 * VoH - 6.98316)
        return F0 + (1.0 - F0) *(2**coeff)
    def _render(self,n,d,r,s,l,v,dis, use_spec=True, isAmbient=False, per_point=False, intensity=None):
        lampIntensity = 1.0
        eps = torch.ones_like(dis)*1e-12
        
        r = r*0.5+0.5
        # r = torch.ones_like(r)*0.6
        d = d*0.5+0.5
        # d = torch.ones_like(d)*0.6
        s = s*0.5+0.5
        # s = torch.ones_like(s)*0.6
        if per_point:
            dim = -2
        else:
            dim = -3
        n = torch_norm(n, dim=dim)
        h = torch_norm((l+v) * 0.5 , dim=dim)

        NoH = torch_dot(n,h, dim=dim)
        NoV = torch_dot(n,v, dim=dim)
        NoL = torch_dot(n,l, dim=dim)
        VoH = torch_dot(v,h, dim=dim)

        NoH = torch.max(NoH, eps)
        NoV = torch.max(NoV, eps)
        NoL = torch.max(NoL, eps)
        VoH = torch.max(VoH, eps)

        f_d = d * self.INV_PI

        D = self.GGX(NoH,r)
        G = self.SmithG(NoV, NoL, r)
        F = self.Fresnel(s, VoH)
        f_s = D * G * F / (4.0 * NoL * NoV + self.EPS)

        if use_spec:
            res =  (f_d + f_s) * NoL * math.pi/dis
        else:
            res = (f_d) * NoL * math.pi /dis
        if intensity is None:
            lampIntensity = self.get_intensity(isAmbient,self.useAugmentation, self.nbRendering)
            if isinstance(lampIntensity,torch.Tensor):
                shape = [res.shape[s] if s == 0 else 1 for s in range(len(res.shape))]
                res = res * lampIntensity.view(*shape)
        else: 
            lampIntensity = intensity
        res *= lampIntensity
        # res = torch.clip(res, 0, 1)
        if self.nbRendering == 1:
            res.squeeze_(0)
        return res
    
    def get_intensity(self, isAmbient=False, useAugmentation=False, ln=1):
        if not isAmbient:
            if useAugmentation:
                #The augmentations will allow different light power and exposures                 
                stdDevWholeBatch = torch.exp(torch.randn(()).normal_(mean = -2.0, std = 0.5))
                #add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy.
                lampIntensity = torch.abs(torch.randn((ln)).normal_(mean = 1.5, std = stdDevWholeBatch)) # Creates a different lighting condition for each shot of the nbRenderings Check for over exposure in renderings
                #autoExposure
                autoExposure = torch.exp(torch.randn(()).normal_(mean = np.log(1), std = 0.4))
                lampIntensity = lampIntensity * autoExposure
            elif hasattr(self, 'lampIntensity'):
                lampIntensity = self.lampIntensity
            else:
                lampIntensity = 3.0 #Look at the renderings when not using augmentations
        else:
            #If this uses ambient lighting we use much small light values to not burn everything.
            if useAugmentation:
                lampIntensity = torch.exp(torch.randn(()).normal_(mean = torch.log(0.15), std = 0.5)) #No need to make it change for each rendering.
            else:
                lampIntensity = 0.15
        return lampIntensity
    
    def set_lamp_intensity(self, lamp_intensity):
        self.lampIntensity = lamp_intensity
    
    def reset_lamp_intensity(self):
        if self.opt.get('lampIntensity'):
            self.lampIntensity = self.opt['lampIntensity']
        else:
            del self.lampIntensity

    def get_dirs(self):
        return self.light_dir, self.view_dir, self.light_dis, self.surface
    
    # ---------------------- Directions ---------------------------
    def torch_generate(self, camera_pos_world, light_pos_world, surface = None, pos = None):
        # permute = self.opt['permute_channel']
        size = self.opt['size']
        nl,_ = light_pos_world.shape
        nv,_ = camera_pos_world.shape
        if pos is None and surface is None:
            pos = self.generate_surface(size)
        elif surface is not None:
            pos = surface
        else:
            pos.unsqueeze_(-1).unsqueeze_(-1)

        light_pos_world = light_pos_world.view(nl,1,1,3)
        camera_pos_world = camera_pos_world.view(nv,1,1,3)

        light_pos_world = light_pos_world.permute(0,3,1,2).contiguous()
        camera_pos_world = camera_pos_world.permute(0,3,1,2).contiguous()

        view_dir_world = torch_norm(camera_pos_world - pos, dim=1)

        # pos = torch.tile(pos,[n,1,1,1])
        light_dis_square = torch.sum(torch.square(light_pos_world - pos),1,keepdims=True)

        light_dir_world = torch_norm(light_pos_world - pos, dim=1)

        return light_dir_world, view_dir_world, light_dis_square, pos
    
    def fullRandomLightsSurface(self):
        nbRenderings = self.nbRendering
        currentLightPos = torch.rand(nbRenderings, 2) * 2 - 1
        currentLightPos = torch.cat([currentLightPos, torch.ones([nbRenderings, 1])* lightDistance], axis = -1)

        # [batch, n, 3]
        return currentLightPos.float()

    def fullRandomCosine(self, n_diff = 3, n_spec = 6):
        # diffuse
        currentViewPos_diff = self.random_dir_cosine(n = n_diff)
        currentLightPos_diff = self.random_dir_cosine(n = n_diff)

        #specular 
        currentViewPos_spec = self.random_dir_cosine(n = n_spec)
        currentLightPos_spec = currentViewPos_spec * torch.from_numpy(np.array([[-1],[-1],[1]], dtype=np.float32))

        wi = torch.cat([currentLightPos_diff, currentLightPos_spec], dim=1).permute(1,0).contiguous()
        wo = torch.cat([currentViewPos_diff, currentViewPos_spec], dim=1).permute(1,0).contiguous()
        return wi, wo

    def fixedLightsSurface(self):
        nbRenderings = self.nbRendering
        currentLightPos = torch.from_numpy(np.array([0.0, 0.0, lightDistance])).view(1,3)
        currentLightPos = torch.cat([currentLightPos]*nbRenderings, dim=0)
        # [n, 3]
        return currentLightPos.float()

    def fixedView(self):
        currentViewPos = torch.from_numpy(np.array([0.0, 0.0, viewDistance])).view(1,3)
        # [1,1,3]
        return currentViewPos.float()

    def generate_surface(self, size):
        x_range = torch.linspace(-1,1,size)
        y_range = torch.linspace(-1,1,size)
        y_mat, x_mat = torch.meshgrid(x_range, y_range, indexing='ij')
        pos = torch.stack([x_mat, -y_mat, torch.zeros(x_mat.shape)],axis=0)
        pos = torch.unsqueeze(pos,0)
        return pos

    def shuffle_sample(self, max_num, num, n, ob, l, v, p, d, mask):
        idx = np.random.permutation(np.arange(n.shape[-1]*n.shape[-2]-torch.sum(mask).numpy()))
        arr = [n.unsqueeze(0),ob.unsqueeze(0),l,v,p[:,:2],d]
        c = 0
        samples = []
        for entity in arr:
            c +=entity.shape[1]
            samples.append(entity[0])
        tmp = torch.cat(samples, dim=0)
        tmp = tmp.masked_select(~mask).view(tmp.shape[0], -1)[:,idx[:num]].numpy().T
        if max_num is not None:
            trace = np.zeros((max_num, c))
            trace[:num, :] = tmp
            idx = np.random.permutation(np.arange(max_num))
            trace = trace[idx, :]
            mask = idx < num
        else:
            trace = tmp
            mask = idx[:num] > -1
        return trace.astype(np.float32), mask
        
    def plane_normal(self, batch = 0):
        size = self.opt['size']
        pos = torch.zeros((2,size,size),dtype=torch.float32)
        z_mat = torch.ones((1,size,size),dtype=torch.float32)
        pos = torch_norm(torch.cat([pos, z_mat],axis=0),dim=0)
        if batch > 0:
            pos = pos.unsqueeze(0).repeat(batch,1,1,1)
        return pos

    def sphere_normal(self, batch = 0, padding=False):
        size = self.opt['size']
        r = 1
        x_range = torch.linspace(-1,1,size)
        y_range = torch.linspace(-1,1,size)
        y_mat, x_mat = torch.meshgrid(x_range, y_range, indexing='ij')
        pos = torch.stack([x_mat, -y_mat],axis=0)
        z_mat = torch.maximum(1-pos[0:1]**2-pos[1:]**2, torch.zeros_like(pos[1:]))
        normal = torch.cat([pos, z_mat],axis=0)
        mask = ~(torch.sum(normal**2, axis=0, keepdim=True) > 1)
        normal = normal * mask
        if padding:
            normal = normal + torch.cat([torch.zeros((2,size,size)), (~mask).float()],dim=0)
        normal = torch_norm(normal,dim=0)
        if batch > 0:
            normal = normal.unsqueeze_(0).repeat(batch,1,1,1)
            mask.unsqueeze_(0)
        return normal, mask
   
    def random_dir(self, batch = 0, n = 1, theta_range=[0,70], phi_range=[0,360]):
        if batch == 0:
            shape = (1, n)
        else:
            shape = (batch, 1, n)
        theta = torch.empty(shape, dtype=torch.float32).uniform_(theta_range[0]/180*np.pi,theta_range[1]/180*np.pi)
        phi = torch.empty(shape, dtype=torch.float32).uniform_(phi_range[0]/180*np.pi,phi_range[1]/180*np.pi)

        x = torch.sin(theta)*torch.cos(phi)
        y = torch.sin(theta)*torch.sin(phi)
        z = torch.cos(theta)
        return torch.cat([x,y,z],dim=-2)
    def random_dir_cosine(self, batch = 0, n = 1, lowEps = 0.001, highEps =0.05):
        if batch == 0:
            shape = (1, n)
        else:
            shape = (batch, 1, n)
        r1 = torch.empty(shape, dtype=torch.float32).uniform_(0.0 + lowEps, 1.0 - highEps)
        r2 = torch.empty(shape, dtype=torch.float32).uniform_(0.0, 1.0)
        r = torch.sqrt(r1)
        phi = 2 * math.pi * r2
        
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.sqrt(1.0 - torch.square(r))
        finalVec = torch.cat([x, y, z], dim=-2) #Dimension here is [batchSize?, 3, n]
        return finalVec
    
    def random_pos(self, batch = 0, n = 1, r=[-1,1]):
        if batch == 0:
            shape = (2, n)
        else:
            shape = (batch, 2, n)
        xy = torch.empty(shape, dtype=torch.float32).uniform_(r[0],r[1])
        if batch == 0:
            shape = (1, n)
        else:
            shape = (batch, 1, n)
        z = torch.zeros(shape, dtype=torch.float32)
        return torch.cat([xy,z], dim=-2)

class Directions():
    def __init__(self):
        pass

    def ldRandom(self):
        pass

class Lighting:
    def __init__(self, opt):
        self.nbRendering = opt['nbRendering']
        self.useAugmentation = opt.get('useAug',False)
        if opt.get('lampIntensity', False):
            self.lampIntensity = opt['lampIntensity']

class RectLighting(Lighting):
    def __init__(self):
        super().__init__()
    
    def sample(self):
        pass
    
    def fetchTex(self):
        pass
    
    def fetchTexConv(self):
        pass

class PolyRender(Render):
    def __init__(self, opt):
        super().__init__(opt)
    
    def render(self):
        pass
    
    def fetchM(self):
        pass
        
    def clipLight(self):
        pass

def rand_n(range_theta, range_phi, nb):
    # return normalized n (nb, 3)
    theta = np.random.uniform((range_theta[0]/180*np.pi), (range_theta[1]/180*np.pi),(nb,1))
    phi = np.random.uniform((range_phi[0]/180*np.pi), (range_phi[1]/180*np.pi),(nb,1))
    return np.concatenate([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1).astype(np.float32)

def random_tangent(n):
    theta_range = [0,180]
    phi_range = [0,360]
    theta = np.random.uniform(theta_range[0]/180*np.pi,theta_range[1]/180*np.pi)
    phi =  np.random.uniform(phi_range[0]/180*np.pi,phi_range[1]/180*np.pi)

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    bi = np.array([x,y,z],dtype=np.float32)
    t = numpy_norm(np.cross(n,bi))

    return t

def log_normalization(img, eps=1e-2):
    return (torch.log(img+eps)-torch.log(torch.ones((1,))*eps))/(torch.log(1+torch.ones((1,))*eps)-torch.log(torch.ones((1,))*eps))

if __name__=="__main__":
    pass