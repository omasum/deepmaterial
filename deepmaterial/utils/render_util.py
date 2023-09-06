#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from typing import List
import numpy as np
import torch
from deepmaterial.utils.ltc_util import LTC
from deepmaterial.utils.vector_util import numpy_norm, torch_cross, torch_norm, torch_dot
from deepmaterial.utils.img_util import gaussianBlurConv, img2tensor, toLDR_torch, toHDR_torch, preprocess
from abc import ABCMeta, abstractmethod
import cv2 as cv
from torch.nn import functional as F
from deepmaterial.utils.wrapper_util import timmer
import nvdiffrast.torch as dr

lightDistance = 2.14
# lightDistance = 1.14
viewDistance = 5  # 44 degrees FOV 


class BRDF(metaclass=ABCMeta):
    '''
    The base class of brdf, containing several brdf term implement (ggx, ...).
    Every subclass should implement the abstractmethod 'eval(), sample()'.
    '''
    def __init__(self, opt):
        '''
        Construct method

        Args:
            opt (dict): The configuration of current brdf instance, containing none properties.
            The class inherited from this class could pass the parameters through 'opt'.
        '''
        self.opt = opt
        self.brdf = None

    def GGX(self, NoH, roughness, eps=1e-12):
        '''
        Isotropic GGX distribution based on
        "Microfacet Models for Refraction through Rough Surfaces"
        https://www.cs.cornell.edu//~srm/publications/EGSR07-btdf.pdf

        Args:
            NoH (Tensor): the dot product (cosine) of middle vector and surface normal
            roughness (Tensor): roughness of the surface

        Returns:
            Tensor : The evaluation result of given roughness and NoH
        '''
        alpha = roughness * roughness
        eps = torch.ones_like(NoH) * eps
        tmp = alpha / torch.max(eps, (NoH * NoH * (alpha * alpha - 1.0) + 1.0))
        return tmp * tmp * (1 / torch.pi)

    def SmithG(self, NoV, NoL, roughness, eps=1e-12):
        '''
        Smith model of the geometry term in microfacet model

        Args:
            NoV (Tensor): the dot production (cosine) of surface normal and view direction
            NoL (Tensor): the dot production (cosine) of surface normal and light direction
            roughness (Tensor): roughness of the surface
            eps (float, optional): eps. Defaults to 1e-12.
        '''
        def _G1(NoM, k):
            return NoM / (NoM * (1.0 - k) + k)

        if isinstance(roughness, torch.Tensor):
            eps = torch.ones_like(roughness) * eps
            k = torch.max(eps, roughness * roughness * 0.5)
        else:
            k = roughness * roughness * 0.5 if roughness ** 2 * 0.5 > eps else eps
        return _G1(NoL, k) * _G1(NoV, k)

    def Fresnel(self, F0, VoH, method='UE'):
        '''
        The fresnel term of microfact model. The method has two type of implementation controlled by 'method' parameter.

        Args:
            F0 (Tensor): The Fresnel reflectance at 0 degree or the rate of reflectance radiance to irradiance
            VoH (Tensor): The dot production of microface normal (half-vector) and view direction
            method (str, optional): Type of implemented Fresnel equation. Defaults to 'UE'.

        Returns:
            Tensor: Fresnel evaluation result
        '''
        if method == 'schlick':
            coeff = (1 - VoH) ** 5
        else:
            coeff = 2**(VoH * (-5.55473 * VoH - 6.98316))
        return F0 + (1.0 - F0) * coeff

    @abstractmethod
    def eval(self):
        '''
        Abstract method for evaluation the brdf model given the light, view and position configuration.
        '''
        pass

    @abstractmethod
    def sample(self):  # brdf method ['ggx', 'benbeckman', 'billin-phong', etc.]
        '''
        Abstract method for importance sampling from the normal distribution function.
        Wrapper function of the other implemented sampling function, such as sampleGGX().
        '''
        pass

    def sampleGGX(self, u, v, roughness, viewDir):
        '''Important sampling according to ggx distribution
        The posibility function is D*dot(wh, n). wh means the normal of microfacet, and n is the normal of macrofacet

        Args:
            u (n-D tensor): random sample from uniform distribution
            v (n-D tensor): random sample from uniform distribution
            roughness (n-D tensor): roughness that control the GGX distribution
            viewDir (n-D tensor): view vector for calculating sampled incident lighting vector

        Returns:
            L: sampled incident lighting vectors
            pdf: the posibility of sampled vectors
        '''

        alpha = roughness ** 2
        phi = 2 * torch.pi * u
        cosTheta = torch.sqrt((1 - v) / (1 + (alpha**2 - 1) * v))
        sinTheta = torch.sqrt(1 - cosTheta * cosTheta)

        H = torch.stack([sinTheta * torch.cos(phi), sinTheta * torch.sin(phi), cosTheta])
        VoH = torch_dot(H, viewDir, dim=-3)
        L = 2 * VoH * H - viewDir

        d = (cosTheta * alpha**2 - cosTheta) * cosTheta + 1
        D = alpha**2 / (torch.pi * d * d)
        pdf = D * cosTheta / 4.0 / VoH

        return L, pdf

    def setPointSet(self, u, v):
        '''set the sample point set for importantance sampling

        Args:
            u (Tensor): random point set
            v (Tensor): random point set
        '''
        self.u = u
        self.v = v

    def norm(self, thetaView=60, roughness=1.0, nSample=1024, fresNorm=False, fresMode='UE'):
        '''Spherical normalization of brdf represented by arbitrary model

        Args:
            nSample (int, optional): The number of samples for integration. Defaults to 1024.

        Returns:
            float: The result of integration
        '''
        results = self.sphricalIntegrate(
            thetaView=thetaView, roughness=roughness, nSample=nSample, fresNorm=fresNorm, fresMode=fresMode)
        if not fresNorm:
            res, pdf, vec = results
            norm = res / pdf * (pdf > 0)
            return norm.mean()
        else:
            res, pdf, vec, fres = results
            norm = res / pdf * (pdf > 0)
            return norm.mean(), fres

    def sphricalIntegrate(self, thetaView=60, roughness=1.0, nSample=1024, norm=False, avgDir=False, fresNorm=False, fresMode='UE'):
        '''Generate vectors over sphere, and calculating the sphrical integration of the brdf

        Args:
            thetaView (float, optional): the angle of view vector. Defaults to 60.
            roughness (float, optional): roughness. Defaults to 1.0.
            nSample (int, optional): samples of vector for integration. Defaults to 1024.
            norm (bool, optional): whether return the normalization. Defaults to False.
            avgDir (bool, optional): whether return the average direction. Defaults to False.

        Returns:
            res: the evaluation of brdf
            pdf: the pdf of generated vectors
            vec: generated vectors
            norm(optional): normalization of brdf
            avgDir: average direction
        '''
        u = (torch.arange(0, nSample, 1, dtype=torch.float32) + 0.5) / nSample
        v = (torch.arange(0, nSample, 1, dtype=torch.float32) + 0.5) / nSample

        u, v = torch.meshgrid(u, v, indexing='ij')
        self.setPointSet(u, v)
        n = torch.stack([torch.zeros_like(u), torch.zeros_like(u), torch.ones_like(u)], dim=0).unsqueeze(0)
        theta = thetaView / 180 * np.pi
        viewDir = torch.from_numpy(np.array([np.sin(theta), 0, np.cos(theta)], dtype=np.float32)).view(1, 3, 1, 1)
        res, pdf, vec, h = self.eval(n=n, r=roughness, v=viewDir, useDiff=False, useFre=False, importantSampling=True)
        results = (res, pdf, vec)
        if fresNorm:
            VoH = torch_dot(viewDir, h, dim=1)
            if fresMode == 'schlick':
                fres = res * (1 - VoH)**5 / pdf * (pdf > 0)
            else:
                fres = res * 2**(VoH * (-5.55473 * VoH - 6.98316)) / pdf * (pdf > 0)
            results += (fres.mean(),)
        if norm:
            norm = res / pdf * (pdf > 0)
            results += (norm.mean(),)
        if avgDir:
            avgDir = res * vec / pdf * (pdf > 0)
            avgDir = avgDir.mean((0, 2, 3))
            avgDir[2] = 0.0
            results += (torch_norm(avgDir, dim=0),)
        return results

    def avgVec(self, thetaView=60, roughness=1.0, nSample=1024):
        '''Calculating the average dircetion of brdf at perticular view direction and roughness

        Args:
            thetaView (float, optional): view direction. Defaults to 60.
            roughness (float, optional): roughness. Defaults to 1.0.
            nSample (int, optional): number of the smaple. Defaults to 1024.

        Returns:
            avgDir: average direction
        '''
        res, pdf, vec = self.sphricalIntegrate(thetaView=thetaView, roughness=roughness, nSample=nSample)
        avgDir = res * vec / pdf * (pdf > 0)
        avgDir = avgDir.sum((0, 1, 3, 4))
        avgDir[1] = 0.0
        avgDir = torch_norm(avgDir, dim=0)
        return avgDir

    def testSample(self, thetaView=60, roughness=1.0, nSample=1024):
        '''Test the important sampling of ggx.

        Args:
            thetaView (float, optional): view direction. Defaults to 60.
            roughness (float, optional): roughness. Defaults to 1.0.
            nSample (int, optional): number of samples. Defaults to 1024.
        '''
        stepu = torch.pi / 2 / 1024
        stepv = 2 * torch.pi / 1024

        u = torch.arange(0, torch.pi / 2, stepu)
        v = torch.arange(0, 2 * torch.pi, stepv)
        u, v = torch.meshgrid(u, v, indexing='ij')
        n = torch.stack([torch.zeros_like(u), torch.zeros_like(u), torch.ones_like(u)], dim=0)
        vec = torch.stack([torch.sin(u) * torch.cos(v), torch.sin(u) * torch.sin(v), torch.cos(u)], dim=0)
        NoH = torch_dot(n, vec, dim=0)
        pdf = self.GGX(NoH, roughness=roughness) * NoH
        momentGT = (pdf * vec * torch.sin(u) * stepu * stepv).sum((1, 2))
        print('ground truth moment: ', momentGT)

        u = (torch.arange(0, nSample, 1, dtype=torch.float32) + 0.5) / nSample
        v = (torch.arange(0, nSample, 1, dtype=torch.float32) + 0.5) / nSample
        u, v = torch.meshgrid(u, v, indexing='ij')
        self.setPointSet(u, v)
        theta = thetaView / 180 * np.pi
        viewDir = torch.from_numpy(np.array([np.sin(theta), 0, np.cos(theta)], dtype=np.float32)).view(1, 3, 1, 1)
        res, pdf, vec, h = self.eval(n=n, r=roughness, v=viewDir, useDiff=False, useFre=False, importantSampling=True)

        momentEs = h.mean((0, 1, 3, 4))
        print('estimated moment: ', momentEs)

    def testNorm(self):
        print(self.norm(roughness=0.36))
        print(self.avgVec(roughness=0.36))

class PlanarSVBRDF(BRDF):
    '''
    A subclass of BRDF which is a planar svBRDF represented by an multi-channel matrix with the same size of surface matrix.
    '''
    def __init__(self, opt, svbrdf=None):
        '''
        Construction function

        Args:
            opt (dict): Configuration of svbrdf, which contains following properties:
                size: the size of surface and svBRDF matrix
            svbrdf (Tensor, optional): Pre-computed svbrdf. Defaults to None.
        '''
        super().__init__(opt)
        self.size = opt['size']
        self.brdf = svbrdf

    def sample(self, u, v):
        return self.sampleGGX(u, v)

    def split_img(self, imgs, split_num=5, split_axis=1, concat=True, svbrdf_norm=True):
        """Split input image to $split_num images.

        Args:
            imgs (list[ndarray] | ndarray): Images with shape (h, w, c). range(0,1)
            split_num (int): number of input images containing
            split_axis (int): which axis the split images is concated.
            svbrdf_norm(bool): if True, process numpy array from (0,1) to (-1,1), and may do gamma_correct
        Returns:
            list[ndarray]: Split images.
        """
        order = self.opt['order']
        gamma_correct = self.opt.get('gamma_correct', '')

        def _norm(img, gamma_correct='pdrs', order='pndrs'):
            n, d, r, s = None, None, None, None
            if 'ndrs' in order:
                n, d, r, s = img[-4:]
            elif order == 'dnrs':
                d, n, r, s = img
            r = np.mean(r, axis=-1, keepdims=True)
            if 'd' in gamma_correct:
                d = d**2.2
            if 'r' in gamma_correct:
                r = r**2.2
            if 's' in gamma_correct:
                s = s**2.2
            result = []
            if order == 'pndrs' or order == 'ndrs':
                result = [n, d, r, s]
            elif order == 'dnrs':
                result = [d, n, r, s]
            result = [preprocess(x) for x in result]
            return result

        if split_num == 1:
            return imgs
        else:
            if isinstance(imgs, list):
                imglist = [np.split(v, split_num, axis=split_axis) for v in imgs]
                if svbrdf_norm:
                    imglist = [_norm(v, gamma_correct, order) for v in imglist]
                if concat:
                    return [np.concatenate(v, axis=-1) for v in imglist]
            else:
                imglist = np.split(imgs, split_num, axis=split_axis)
                if svbrdf_norm:
                    imglist = _norm(imglist, gamma_correct, order)
                if concat:
                    return np.concatenate(imglist, axis=-1)

    def get_svbrdfs(self, imgs, normalization=False, num=5, axis=1, concat=True, norm=True):
        self.brdf = img2tensor(self.split_img(imgs, split_num=num, split_axis=axis, concat=concat,
                               svbrdf_norm=norm), bgr2rgb=False, float32=True, normalization=normalization)
        return self.brdf
    
    def get_normals(self, imgs, normalization=False, num=7, axis=1, concat=False, norm=True):
        self.brdf = img2tensor(self.split_single(imgs, split_num=num, split_axis=axis, concat=concat,
                               svbrdf_norm=norm), bgr2rgb=False, float32=True, normalization=normalization)
        return self.brdf # [b, 3, h, w]
    
    def split_single(self, imgs, split_num=5, split_axis=1, concat=True, svbrdf_norm=True):
        """Split input image to $split_num images.

        Args:
            imgs (list[ndarray] | ndarray): Images with shape (h, w, c). range(0,1)
            split_num (int): number of input images containing
            split_axis (int): which axis the split images is concated.
            svbrdf_norm(bool): if True, process numpy array from (0,1) to (-1,1), and may do gamma_correct
        Returns:
            list[ndarray]: Split images.
        """

        def _norm(img):
            '''
                img is a list
            '''
            assert len(img) == 7, 'the input is not 7 image'
            gt, pred, band1, band2, band3, band4, band5 = img
            result = preprocess(gt)
            return result # [b,3,h,w]

        if split_num == 1:
            return imgs
        else:
            if isinstance(imgs, list):
                imglist = [np.split(v, split_num, axis=split_axis) for v in imgs]
                if svbrdf_norm:
                    imglist = [_norm(v) for v in imglist]
                if concat:
                    return [np.concatenate(v, axis=-1) for v in imglist]
            else:
                imglist = np.split(imgs, split_num, axis=split_axis) # len(imglist = 7)
                if svbrdf_norm:
                    imglist = _norm(imglist)
                if concat:
                    return np.concatenate(imglist, axis=-1)
                else:
                    return imglist
    

    def setSVBRDF(self, svbrdf):
        self.brdf = svbrdf

    def _seperate_brdf(self, svbrdf=None, n_xy=False, r_single=True):
        if svbrdf is None:
            svbrdf = self.brdf
        if svbrdf.dim() == 4:
            b, c, h, w = svbrdf.shape
        else:
            b = 0
            svbrdf = svbrdf.unsqueeze(0)

        if self.opt['order'] == 'pndrs' or self.opt['order'] == 'ndrs':
            if not n_xy:
                n = svbrdf[:, 0:3]
                d = svbrdf[:, 3:6]
                if r_single:
                    r = svbrdf[:, 6:7]
                    s = svbrdf[:, 7:10]
                else:
                    r = svbrdf[:, 6:7]
                    s = svbrdf[:, 9:12]
            else:
                n = svbrdf[:, 0:2]
                n = self.unsqueeze_normal(n)
                d = svbrdf[:, 2:5]
                if r_single:
                    r = svbrdf[:, 5:6]
                    s = svbrdf[:, 6:9]
                else:
                    r = svbrdf[:, 5:6]
                    s = svbrdf[:, 8:11]
        elif self.opt['order'] == 'dnrs':
            d = svbrdf[:, 0:3]
            n = svbrdf[:, 3:6]
            r = svbrdf[:, 6:7]
            s = svbrdf[:, 7:10]
        if b != 0:
            n = n.unsqueeze(1)
            d = d.unsqueeze(1)
            r = r.unsqueeze(1)
            s = s.unsqueeze(1)
        return torch_norm(n,dim=-3), d, r, s

    def squeeze_brdf(self, svbrdf=None, n_xy=True):
        '''squeeze the normal of svbrdf to two channels

        Args:
            svbrdf (array, optional): svbrdf to squeeze. Defaults to None.
            n_xy (bool, optional): whether normal needs to be squeezed. Defaults to True.

        Returns:
            svbrdf: the squeezed svbrdf
        '''
        if svbrdf is None:
            assert (self.brdf.shape[-3] == 10)
            svbrdf = self.brdf
        n, d, r, s = self._seperate_brdf(svbrdf)
        if n_xy:
            n = PlanarSVBRDF.squeeze_normal(n)
        n = n.squeeze(0)
        d = d.squeeze(0)
        r = r.squeeze(0)
        s = s.squeeze(0)
        self.brdf = torch.cat([n, d, r, s], dim=0)
        return svbrdf

    def unsqueeze_brdf(self, svbrdf=None, n_xy=True, r_single=True):
        '''unsqueeze the normal of svbrdf to three channels

        Args:
            svbrdf (array, optional): svbrdf to unsqueeze. Defaults to None.
            n_xy (bool, optional): whether normal needs to be unsqueezed. Defaults to True.

        Returns:
            svbrdf: the unsqueezed svbrdf
        '''
        if svbrdf is None:
            assert (self.brdf.shape[-3] == 9)
            svbrdf = self.brdf
        n, d, r, s = self._seperate_brdf(svbrdf, n_xy=n_xy, r_single=r_single)
        if n_xy:
            n = PlanarSVBRDF.unsqueeze_normal(n)
        n = n.squeeze(0)
        d = d.squeeze(0)
        r = r.squeeze(0)
        s = s.squeeze(0)
        if r_single:
            svbrdf = torch.cat([n, d, r, r, r, s], dim=0)
        else:
            svbrdf = torch.cat([n, d, r, s], dim=0)
        return svbrdf

    @staticmethod
    def homo2sv(brdf, size=(256, 256)):
        if len(brdf.shape) >= 2:
            n, c = brdf.shape
            svbrdf = torch.ones((n, c) + size, dtype=torch.float32) * (brdf.unsqueeze(-1).unsqueeze(-1))
        else:
            c = brdf.shape[0]
            svbrdf = torch.ones((c,) + size, dtype=torch.float32) * (brdf.unsqueeze(1).unsqueeze(1))
        return svbrdf

    @staticmethod
    def brdf2uint8(svbrdf=None, n_xy=True, r_single=True, gamma_correct=''):
        if not n_xy:
            svbrdf = svbrdf / 2 + 0.5
            n = svbrdf[:, 0:3]
            d = svbrdf[:, 3:6]
            if r_single:
                r = svbrdf[:, 6:7]
                s = svbrdf[:, 7:10]
                r = torch.cat([r] * 3, dim=1)
            else:
                r = svbrdf[:, 6:9]
                s = svbrdf[:, 9:12]
        else:
            n = svbrdf[:, 0:2]
            n = PlanarSVBRDF.unsqueeze_normal(n) / 2 + 0.5
            d = svbrdf[:, 2:5] / 2 + 0.5
            if r_single:
                r = svbrdf[:, 5:6] / 2 + 0.5
                s = svbrdf[:, 6:9] / 2 + 0.5
                r = torch.cat([r] * 3, dim=1)
            else:
                r = svbrdf[:, 5:8] / 2 + 0.5
                s = svbrdf[:, 8:11] / 2 + 0.5
        if 'd' in gamma_correct:
            d = d**0.4545
        if 'r' in gamma_correct:
            r = r**0.4545
        if 's' in gamma_correct:
            s = s**0.4545

        result = torch.cat([n, d, r, s], dim=3)
        return result

    @staticmethod
    def squeeze_normal(n):
        '''
            n: normal, shape: (3, h, w)
            return: svbrdf, shape:( 2,h,w)
        '''
        n = torch_norm(n, 1)[:, :2]
        return n

    @staticmethod
    def unsqueeze_normal(n):
        b, c, h, w = n.shape
        z = torch.ones((b, 1, h, w), dtype=torch.float32) - (n**2).sum(dim=1)
        n = torch.cat([n, z], dim=1)
        n = torch_norm(n, 1)
        return n

    def eval(self, n=None, d=None, r=None, s=None, l=None, v=None, dis=None,
             useDiff=True, useSpec=True, perPoint=False, intensity=None, useFre=True, importantSampling=False, lossRender=False, eps=1e-12):
        c, ih, iw = n.shape[-3:]
        trEps = torch.ones((1, ih, iw)) * eps

        r = torch.clip(r * 0.5 + 0.5, 0.001)
        # r = torch.ones_like(r)*0.6
        if perPoint:
            dim = -2
        else:
            dim = -3

        if importantSampling and l is None:
            phi = 2 * torch.pi * self.v
            cosTheta = torch.sqrt((1 - self.u) / (1 + (r**4 - 1) * self.u))
            sinTheta = torch.sqrt(1 - cosTheta * cosTheta)
            h = torch.stack([sinTheta * torch.cos(phi), sinTheta * torch.sin(phi), cosTheta]).unsqueeze(0).unsqueeze(0)
            h = Directions.worldToTangent(h, n, mode='static', permute=True)
            VoH = torch_dot(v, h, dim=dim)
            l = 2 * VoH * h - v
        else:
            h = torch_norm((l + v) * 0.5, dim=dim)
            VoH = torch_dot(v, h, dim=dim)
        n = torch_norm(n, dim=dim)

        NoH = torch_dot(n, h, dim=dim)
        NoV = torch_dot(n, v, dim=dim)
        NoL = torch_dot(n, l, dim=dim)

        NoH = torch.clip(NoH, eps)
        NoV = torch.clip(NoV, eps)
        NoL = torch.clip(NoL, eps)
        VoH = torch.clip(VoH, eps)

        D = self.GGX(NoH, r)
        G = self.SmithG(NoV, NoL, r)
        if useFre:
            s = s * 0.5 + 0.5
            # s = torch.ones_like(s)*0.6
            F = self.Fresnel(s, VoH)
        else:
            F = 1.0
        res = 0
        if useDiff:
            d = d * 0.5 + 0.5
            # d = torch.ones_like(d)*0.6
            f_d = d * (1-s) * (1 / torch.pi)
            res += f_d * NoL
        if useSpec:
            f_s = D * G * F / 4.0
            if not lossRender:
                f_s /= (NoL * NoV + eps)
            res += f_s * NoL
        if intensity is not None:
            res *= intensity / dis
        if importantSampling:
            pdf = D * NoH / 4.0 / VoH
            return res, pdf, l, h
        res = torch.clip(res, 0.0, 1.0)
        return res

    def to(self, device):
        self.brdf = self.brdf.to(device)

class Render():
    def __init__(self, opt, lighting = None, device='cpu'):
        self.opt = opt
        self.nbRendering = opt['nbRendering']
        self.size = opt['size']
        if lighting is None:
            self.lighting = Lighting(opt=opt)
        else:
            self.lighting = lighting
        self.dirUtil = Directions(opt=opt)
        self.camera = Camera(opt=opt)
        self.device = device

    def render(self, svbrdf, single_pos=False, obj_pos=None, light_pos=None, view_pos=None, keep_dirs=False, load_dirs=False, light_dir=None,
               view_dir=None, light_dis=None, surface=None, random_light=True, colocated=False, lossRender=False,
               n_xy=False, r_single=True, toLDR=None, useDiff=True, perPoint=False, isAmbient=False):
        '''
        Args:
            light_dir(tensor[3]):[x,y,z](tensor.float32): parallel light direction
        '''
        assert not (light_dir is not None and light_pos is not None), (
            'Given two type of lighting initilization, please choose one type (light_dir, light_pos)')
        assert not (view_dir is not None and view_pos is not None), (
            'Given two type of view initilization, please choose one type (view_dir, view_pos)')
        if not isinstance(svbrdf, PlanarSVBRDF):
            svbrdf = PlanarSVBRDF(self.opt, svbrdf)
        if light_dir is not None or light_pos is not None:
            random_light = False
        if toLDR is None:
            toLDR = self.opt['toLDR']
        if light_dir is None:
            # pointlight set
            if random_light:
                light_pos = self.lighting.fullRandomLightsSurface()
            elif light_pos is not None:
                light_pos = torch.from_numpy(np.array(light_pos)).view(1, 3)
            else:
                light_pos = self.lighting.fixedLightsSurface()
            if colocated:
                view_pos = light_pos
            elif view_pos is not None:
                view_pos = torch.from_numpy(np.array(view_pos)).view(1, 3)
            else:
                view_pos = self.camera.fixedView()

            if obj_pos is None and single_pos:
                obj_pos = torch.rand((self.nbRendering, 3), dtype=torch.float32) * 2 - 1
                obj_pos[:, 2] = 0.0
            light_dir, view_dir, light_dis, surface = self.torch_generate(view_pos, light_pos, pos=obj_pos)
        else:
            # parallel light set
            light_dir = light_dir.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            light_dir = light_dir.repeat(1, 1 ,self.opt['size'], self.opt['size'])
            if colocated:
                view_pos = light_pos
            elif view_pos is not None:
                view_pos = torch.from_numpy(np.array(view_pos)).view(1, 3)
            else:
                view_pos = self.camera.fixedView() # the center of material plane
            light_dir, view_dir, light_dis, surface = self.torch_generate_parallel(view_pos, light_dir, pos=obj_pos)
    
        if keep_dirs:
            self.light_dir = light_dir
            self.view_dir = view_dir
            self.light_dis = light_dis
            self.surface = surface
        if load_dirs:
            light_dir, view_dir, light_dis, surface = self.light_dir, self.view_dir, self.light_dis, self.surface
        n, d, r, s = svbrdf._seperate_brdf(n_xy=n_xy, r_single=r_single)
        lampIntensity = self.lighting.get_intensity(isAmbient, self.opt.get('useAug', False), self.nbRendering)
        if isinstance(lampIntensity, torch.Tensor):
            if len(n.shape) == 3:
                shape = [self.nbRendering, 1, 1, 1]
            if len(n.shape) == 4:
                shape = [n.shape[0], self.nbRendering, 1, 1, 1]
            lampIntensity = lampIntensity.view(*shape)
        render_result = svbrdf.eval(n, d, r, s, light_dir, view_dir, light_dis,
                                    useDiff=useDiff, perPoint=perPoint, intensity=lampIntensity, lossRender=lossRender)
        # res = torch.clip(res, 0, 1)
        if self.nbRendering == 1:
            render_result.squeeze_(0)
        if toLDR:
            render_result = toLDR_torch(render_result)
            render_result = toHDR_torch(render_result)
        return render_result

    def render_panel_single_point(self, brdf, l=None, wi=None, dis=None, wo=None, n_xy=False,
                                  r_single=True, toLDR=None):
        # brdf (9/10, 1)
        # l is a panel lighting matrix, in which the element is the intensity of lighting panel.
        # l: (1,hl,wl)
        # wi is a panel lighting matrix, in which the element is direction from the sample point in lighting panel to surface point.
        # if wi is given, the distance need to be given too.
        # wi: (3,hl,wl)
        # wo: (3,ho,ho)
        # dis: (1,hl,wl)
        if not isinstance(svbrdf, PlanarSVBRDF):
            svbrdf = PlanarSVBRDF(self.opt, svbrdf)
        if toLDR is None:
            toLDR = self.opt['toLDR']
        h, w = wo.shape[-2:]
        wi = wi.permute(1, 0).contiguous().view(1, -1, 3, 1, 1).repeat(h * w, 1, 1, 1, 1)
        l = l.permute(1, 0).contiguous().view(1, -1, 3, 1, 1).repeat(h * w, 1, 1, 1, 1)
        wo = wo.permute(1, 2, 0).contiguous().view(-1, 1, 3, 1, 1).repeat(1, l.shape[1], 1, 1, 1)
        if dis is not None:
            dis = dis.permute(1, 2, 0).contiguous().view(1, -1, 1, 1, 1).repeat(h * w, 1, 1, 1, 1)
            l = l / dis**2
        else:
            dis = torch.ones_like(wi)
        v = wo
        n, d, r, s = brdf._seperate_brdf(n_xy=n_xy, r_single=r_single)
        n.unsqueeze_(0)
        d.unsqueeze_(0)
        r.unsqueeze_(0)
        s.unsqueeze_(0)
        render_result = brdf.eval(n, d, r, s, wi, v, dis, perPoint=False, intensity=l)
        # res = torch.clip(res, 0, 1)
        if self.nbRendering == 1:
            render_result.squeeze_(0)
        if toLDR:
            render_result = toLDR_torch(render_result)
            render_result = toHDR_torch(render_result)
        render_result = render_result.sum(1).view(h, w, 3).permute(2, 0, 1).contiguous()
        return render_result

    def get_dirs(self):
        return self.light_dir, self.view_dir, self.light_dis, self.surface

    def torch_generate(self, camera_pos_world, light_pos_world, surface=None, pos=None, normLight=True):
        '''
            return light_dir_world.shape=[nbRendering,3,size,size]
        '''
        # permute = self.opt['permute_channel']
        size = self.opt['size'] #256
        nl, _ = light_pos_world.shape[-2:]
        nv, _ = camera_pos_world.shape[-2:]
        if pos is None and surface is None:
            pos = self.generate_surface(size)
        elif surface is not None:
            pos = surface
        else:
            pos.unsqueeze_(-1).unsqueeze_(-1)

        light_pos_world = light_pos_world.view(*light_pos_world.shape, 1, 1)
        camera_pos_world = camera_pos_world.view(*camera_pos_world.shape, 1, 1)

        view_dir_world = torch_norm(camera_pos_world - pos, dim=-3)

        # pos = torch.tile(pos,[n,1,1,1])
        light_dis_square = torch.sum(torch.square(light_pos_world - pos), -3, keepdims=True)
        # [n,1,256,256]

        if normLight:
            light_dir_world = torch_norm(light_pos_world - pos, dim=-3)
        else:
            light_dir_world = light_pos_world-pos

        return light_dir_world, view_dir_world, light_dis_square, pos

    def torch_generate_parallel(self, camera_pos_world, light_dir, surface=None, pos=None, normLight=True):
        
        # permute = self.opt['permute_channel']
        size = self.opt['size'] #256
        nv, _ = camera_pos_world.shape[-2:]
        if pos is None and surface is None:
            pos = self.generate_surface(size)
        elif surface is not None:
            pos = surface
        else:
            pos.unsqueeze_(-1).unsqueeze_(-1)

        if normLight:
            light_dir_world = torch_norm(light_dir, dim=-3)

        camera_pos_world = camera_pos_world.view(*camera_pos_world.shape, 1, 1)
        view_dir_world = torch_norm(camera_pos_world - pos, dim=-3)

        # pos = torch.tile(pos,[n,1,1,1])
        light_dis_square = torch.ones(self.opt['nbRendering'], 1, size, size)

        return light_dir_world, view_dir_world, light_dis_square, pos

    def generate_surface(self, size):
        '''generate a plane surface with the size of $size

        Args:
            size (int): the size of edge length of surface

        Returns:
            pos: the position array of surface
        '''
        x_range = torch.linspace(-1, 1, size)
        y_range = torch.linspace(-1, 1, size)
        y_mat, x_mat = torch.meshgrid(x_range, y_range, indexing='ij')
        pos = torch.stack([x_mat, -y_mat, torch.zeros(x_mat.shape)], axis=0)
        pos = torch.unsqueeze(pos, 0)
        return pos.to(self.device)

    def fullRandomCosine(self, batch=1, n_diff=3, n_spec=6):
        # diffuse
        if n_diff != 0:
            currentViewPos_diff = self.dirUtil.random_dir_cosine(batch = batch,n=n_diff)
            currentLightPos_diff = self.dirUtil.random_dir_cosine(batch = batch,n=n_diff)

        # specular
        if n_spec != 0:
            posShift = torch.empty(batch, 2, n_spec).uniform_(-1.0,1.0)
            posShift = torch.cat([posShift, torch.zeros([batch, 1, n_spec], dtype=torch.float32)+0.0001], axis=-2)
            distance = torch.empty(batch, 1, n_spec).normal_(0.5, 0.75)
            currentViewPos_spec = self.dirUtil.random_dir_cosine(batch = batch,n=n_spec)
            currentLightPos_spec = currentViewPos_spec * torch.from_numpy(np.array([[-1], [-1], [1]], dtype=np.float32))
            currentViewPos_spec = currentViewPos_spec * torch.exp(distance) + posShift
            currentLightPos_spec = currentLightPos_spec * torch.exp(distance) + posShift
        
        if n_diff != 0 and n_spec != 0:
            wi = torch.cat([currentLightPos_diff, currentLightPos_spec], dim=1).transpose(-2, -1).contiguous()
            wo = torch.cat([currentViewPos_diff, currentViewPos_spec], dim=1).transpose(-2, -1).contiguous()
        elif n_diff!=0:
            wi = currentLightPos_diff.transpose(-2, -1).contiguous()
            wo = currentViewPos_diff.transpose(-2, -1).contiguous()
        else:
            wi = currentLightPos_spec.transpose(-2, -1).contiguous()
            wo = currentViewPos_spec.transpose(-2, -1).contiguous()
        return wi, wo

    def shuffle_sample(self, max_num, num, n, ob, l, v, p, d, mask):
        idx = np.random.permutation(np.arange(n.shape[-1] * n.shape[-2] - torch.sum(mask).numpy()))
        arr = [n.unsqueeze(0), ob.unsqueeze(0), l, v, p[:, :2], d]
        c = 0
        samples = []
        for entity in arr:
            c += entity.shape[1]
            samples.append(entity[0])
        tmp = torch.cat(samples, dim=0)
        tmp = tmp.masked_select(~mask).view(tmp.shape[0], -1)[:, idx[:num]].numpy().T
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

    def plane_normal(self, batch=0):
        '''generate a normal of plane surface with the size of $size

        Args:
            size (int): the size of edge length of normal map

        Returns:
            pos: the normal map of plane surface
        '''
        size = self.opt['size']
        pos = torch.zeros((2, size, size), dtype=torch.float32)
        z_mat = torch.ones((1, size, size), dtype=torch.float32)
        pos = torch_norm(torch.cat([pos, z_mat], axis=0), dim=0)
        if batch > 0:
            pos = pos.unsqueeze(0).repeat(batch, 1, 1, 1)
        return pos

    def sphere_normal(self, batch=0, padding=False):
        '''generate a normal of sphere surface with the size of $size

        Args:
            size (int): the size of edge length of normal map
            padding(bool): whether padding plane normal to the element out of the sphere

        Returns:
            pos: the normal map of sphere surface
        '''
        size = self.opt['size']
        r = 1
        x_range = torch.linspace(-1, 1, size)
        y_range = torch.linspace(-1, 1, size)
        y_mat, x_mat = torch.meshgrid(x_range, y_range, indexing='ij')
        pos = torch.stack([x_mat, -y_mat], axis=0)
        z_mat = torch.maximum(1 - pos[0:1]**2 - pos[1:]**2, torch.zeros_like(pos[1:]))
        normal = torch.cat([pos, z_mat], axis=0)
        mask = ~(torch.sum(normal**2, axis=0, keepdim=True) > 1)
        normal = normal * mask
        if padding:
            normal = normal + torch.cat([torch.zeros((2, size, size)), (~mask).float()], dim=0)
        normal = torch_norm(normal, dim=0)
        if batch > 0:
            normal = normal.unsqueeze_(0).repeat(batch, 1, 1, 1)
            mask.unsqueeze_(0)
        return normal, mask

class Directions():
    def __init__(self, opt):
        self.opt = opt
        pass

    def ldRandom(self):
        pass

    def random_dir(self, batch=0, n=1, theta_range=[0, 70], phi_range=[0, 360]):
        if batch == 0:
            shape = (1, n)
        else:
            shape = (batch, 1, n)
        theta = torch.empty(shape, dtype=torch.float32).uniform_(
            theta_range[0] / 180 * np.pi, theta_range[1] / 180 * np.pi)
        phi = torch.empty(shape, dtype=torch.float32).uniform_(phi_range[0] / 180 * np.pi, phi_range[1] / 180 * np.pi)

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.cat([x, y, z], dim=-2)

    def random_dir_cosine(self, batch=0, n=1, lowEps=0.001, highEps=0.05):
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
        finalVec = torch.cat([x, y, z], dim=-2)  # Dimension here is [batchSize?, 3, n]
        return finalVec

    def random_pos(self, batch=0, n=1, r=[-1, 1]):
        if batch == 0:
            shape = (2, n)
        else:
            shape = (batch, 2, n)
        xy = torch.empty(shape, dtype=torch.float32).uniform_(r[0], r[1])
        if batch == 0:
            shape = (1, n)
        else:
            shape = (batch, 1, n)
        z = torch.zeros(shape, dtype=torch.float32)
        return torch.cat([xy, z], dim=-2)
    
    @staticmethod
    def buildTBNStatic(n):
        judgement = ~torch.logical_or((n[...,2:, :, :] == 1), (n[...,2:, :, :] == -1))
        tZ = torch.stack([torch.zeros_like(n[...,2, :, :]),torch.zeros_like(n[...,2, :, :]),torch.ones_like(n[...,2, :, :])],dim=-3)
        tX = torch.stack([torch.ones_like(n[...,2, :, :]),torch.zeros_like(n[...,2, :, :]),torch.zeros_like(n[...,2, :, :])],dim=-3)
        t = torch.where(judgement, tZ, tX)
        t = torch_norm(t - n * torch_dot(n, t), dim=-3)
        b = torch_cross(n, t, dim=-3)
        TBN = torch.stack([t, b, n], dim=-3)
        return TBN

    @staticmethod
    def worldToTangent(vec, n, v=None, mode='view', permute=False):
        expand=False
        if n.ndim == 3:
            n = n.unsqueeze(0)
        elif n.ndim == 5:
            n = n.squeeze(1)
        if mode == 'static':
            TBN = Directions.buildTBNStatic(n)
        elif mode == 'view':
            TBN = Directions.buildTBNView(n,v)
        if isinstance(vec, List):
            result = []
            for vv in vec:
                tmpV = torch.matmul(TBN.permute(0, 3, 4, 1, 2).contiguous(), vv.permute(
                    2,3,1,0).contiguous()).squeeze(0)
                result.append(tmpV.permute(3, 2, 0, 1).contiguous())
        else:
            # TBN n, 3, 3, h, w
            # vec ln, c, h, w
            vec = vec.permute(0, 3, 4, 2, 1).contiguous()
            if mode == 'static':
                TBN = TBN.permute(0, 3, 4, 1, 2).contiguous()
            result = torch.matmul(TBN, vec)
            if permute:
                result = result.permute(0, 4, 3, 1, 2).contiguous()
        return result

    @staticmethod
    def buildTBNView(n, v):
        NoV = torch_dot(n,v, dim=-1)
        tv = torch_norm(v-n*NoV, dim=-1)
        tx = torch.stack([torch.ones_like(n[:, :, :, 2]),torch.zeros_like(n[:, :, :, 2]),torch.zeros_like(n[:, :, :, 2])],dim=-1)

        t = torch.where(~(NoV == 0.0), tv, tx)
        b = torch_cross(n, t, dim=-1)
        TBN = torch.stack([t, b, n], dim=-2)
        return TBN

class Camera:
    def __init__(self, opt):
        self.opt = opt
        pass

    def fixedView(self):
        currentViewPos = torch.from_numpy(np.array([0.0, 0.0, viewDistance])).view(1, 3)
        # [1,1,3]
        return currentViewPos.float()

class Lighting:
    def __init__(self, opt):
        self.opt = opt
        self.nbRendering = opt['nbRendering']
        self.useAugmentation = opt.get('useAug', False)
        if opt.get('lampIntensity', False):
            self.lampIntensity = opt['lampIntensity']

    def get_intensity(self, isAmbient=False, useAugmentation=False, ln=1):
        if not isAmbient:
            if useAugmentation:
                # The augmentations will allow different light power and exposures
                stdDevWholeBatch = torch.exp(torch.randn(()).normal_(mean=-2.0, std=0.5))
                # add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy.
                # Creates a different lighting condition for each shot of the nbRenderings Check for over exposure in renderings
                lampIntensity = torch.abs(torch.randn((ln)).normal_(mean=1.5, std=stdDevWholeBatch))
                # autoExposure
                autoExposure = torch.exp(torch.randn(()).normal_(mean=np.log(1), std=0.4))
                lampIntensity = lampIntensity * autoExposure
            elif hasattr(self, 'lampIntensity'):
                lampIntensity = self.lampIntensity
            else:
                lampIntensity = 3.0 * np.pi  # Look at the renderings when not using augmentations
        else:
            # If this uses ambient lighting we use much small light values to not burn everything.
            if useAugmentation:
                # No need to make it change for each rendering.
                lampIntensity = torch.exp(torch.randn(()).normal_(mean=torch.log(0.15), std=0.5))
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

    def fullRandomLightsSurface(self, batch=None, lightDistance = lightDistance):
        if batch is None:
            nbRenderings = self.nbRendering
        else:
            nbRenderings = batch
        currentLightPos = torch.rand(nbRenderings, 2) * 2 - 1
        currentLightPos = torch.cat([currentLightPos, torch.ones([nbRenderings, 1]) * lightDistance], axis=-1)

        # [batch, n, 3]
        return currentLightPos.float()

    def fixedLightsSurface(self):
        '''
            generate pointlight with position[0, 0, lightDistance], the shape is [nbRenderings, 3]
        '''
        nbRenderings = self.nbRendering
        currentLightPos = torch.from_numpy(np.array([0.0, 0.0, lightDistance])).view(1, 3)
        currentLightPos = torch.cat([currentLightPos] * nbRenderings, dim=0)
        # [nbRenderings, 3]
        return currentLightPos.float()

class Ray:
    def __init__(self, origin=None, vec=[0,0,1]):
        if origin is None:
            origin = torch.zeros_like(vec)
        self.origin = origin
        self.dir = torch_norm(vec,dim=-3)

class RectLighting(Lighting):
    def __init__(self, opt, center=[0, 0, 2.75], dirx=[1, 0, 0], diry=[0, 1, 0], halfx=2.0, halfy=2.0, device='cpu'):
        super().__init__(opt=opt)
        self.device = device
        self.center = torch.from_numpy(np.array(center, dtype=np.float32)).to(self.device)
        self.dirx = torch.from_numpy(np.array(dirx, dtype=np.float32)).to(self.device)
        self.diry = torch.from_numpy(np.array(diry, dtype=np.float32)).to(self.device)
        self.halfx = halfx
        self.halfy = halfy

        self.vertices = torch.zeros(4, 3, dtype=torch.float32)
        self.initPoints()
        self.initRect()
        texture = opt.get('texture', None)
        textureMode = opt.get('textureMode', 'OpenCV')
        self.nLod = opt.get('nLod', 10)
        self.fetchMode = opt.get('fetchMode', 'Torch')
        if textureMode == 'Torch':
            self.gConvs = self.initGaussianConv(self.nLod)
        if texture is not None:
            self.initTexture(texture, textureMode)
        else:
            self.tex = None

    def initTexture(self, texture, textureMode):
        if texture is not None:
            if isinstance(texture, str):
                self.tex = cv.imread(texture)
            elif isinstance(texture, np.ndarray):
                self.tex = texture
            if textureMode == 'Torch':
                if isinstance(texture, torch.Tensor):
                    self.tex = texture
                else:
                    self.tex = img2tensor(self.tex, normalization=True).to(self.device)
            else:
                self.tex = (self.tex[:,:,::-1]/255).astype(np.float32)
            self.convTex(textureMode, self.nLod)

    def initPoints(self):
        ex = self.dirx * self.halfx
        ey = self.diry * self.halfy

        self.vertices[0,:] = self.center - ex - ey
        self.vertices[1,:] = self.center + ex - ey
        self.vertices[2,:] = self.center + ex + ey
        self.vertices[3,:] = self.center - ex + ey

    def initRect(self):
        self.normal = torch.cross(self.diry, self.dirx).to(self.device)

        # * The plane is a four-dimension vector which store the normal (three dimision) of the plane and the distance (scalar) from the origin to the plane.
        # * The representation of the plane is n.x * X + n.y * Y + n[..., 2] * Z + d = 0
        self.plane = torch.cat([self.normal, -1 * torch_dot(self.normal, self.center, dim=0)], dim=0).to(self.device)

    @staticmethod
    def generateGradientLighting(direction = 'x', size=(256,256)):
        x = torch.linspace(-1, 1, size[0])
        y = torch.linspace(-1, 1, size[1])
        x, y = torch.meshgrid(x, y, indexing='ij')

        if direction == 'x':
            return y/2+0.5
        elif direction == 'y':
            return x/2+0.5
        elif direction == 'z':
            z = torch.sqrt(2 - x**2 - y**2)/np.sqrt(2)
            return z

    def setTangentLighting(self, lighting, normal):
        self.tangentLighting = lighting
        self.tangentNormal = normal
        self.tangentPlane = torch.cat([normal, -torch_dot(lighting[:, 0], normal)], dim=1)

    def intersect(self, ray):
        # * The distance of a point to the plane is n.x * p.x + n.y * p.y + n[..., 2] * p[..., 2] + d (numerator).
        # * The denominator means whether the direction of ray is same as the normal of plane.

        t = - torch_dot(self.tangentPlane, torch.cat([ray.origin, torch.ones_like(ray.origin[:, 0:1])], dim=-3),
                      dim=-3) / torch_dot(self.tangentNormal, ray.dir, dim=-3)
        return t > 0, t

    def sample(self):
        pass

    def initGaussianConv(self, lod=10, kSize=21):
        gConvs = []
        for i in range(lod):
            sigma = np.minimum(np.exp(i)/2, 20.0)
            gConvs.append(gaussianBlurConv(sigma, kSize).float().to(self.device))
        self.gConvs = gConvs
        return gConvs

    def convTex(self, mode='Torch', lod=10):
        if mode=='Torch' and self.tex.ndim == 3:
            tex = self.tex.unsqueeze(0)
        else:
            tex = self.tex
        if self.fetchMode=='Torch':
            self.lod = [tex.squeeze(0)]
        else:
            self.lod = []
        for i in range(lod):
            if mode == 'OpenCV':
                # kSize = 15 * (2**i)
                kSize = 21
                if kSize % 2 == 0:
                    kSize += 1 
                tex = cv.resize(tex, dsize=(0,0), fx = 0.5, fy = 0.5,interpolation=cv.INTER_LINEAR)
                consOne = np.ones_like(tex)
                img = cv.GaussianBlur(tex, ksize=(kSize,kSize), sigmaX = np.minimum(np.exp(i)/2, 9.0), borderType=cv.BORDER_CONSTANT)
                img /= cv.GaussianBlur(consOne, ksize=(kSize,kSize), sigmaX = np.minimum(np.exp(i)/2, 9.0), borderType=cv.BORDER_CONSTANT)
                img = cv.resize(img, dsize=self.tex.shape[:2],interpolation=cv.INTER_LINEAR)
                self.lod.append(torch.from_numpy(img).permute(2,0,1).contiguous())
            elif mode == 'Torch':
                #! the shape of self.tex is [n, 3, h, w]
                tex = F.interpolate(tex, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
                consOne = torch.ones_like(tex)
                img = self.gConvs[i](tex)
                img = img / self.gConvs[i](consOne)
                if self.fetchMode == 'Torch':
                    img = F.interpolate(img,(self.tex.shape[-2:]), mode='bilinear',align_corners=False)
                    self.lod.append(img.squeeze(0).contiguous())
                else:
                    self.lod.append(img.permute(0,2,3,1).contiguous())
        if self.fetchMode == 'Torch':
            self.lod = torch.stack(self.lod, dim=0)

    def setTex(self, texture):
        self.tex = texture

    def lodToDevice(self, device):
        self.lod = self.lod.to(device)
    
    def verticesToDevice(self, device):
        self.vertices = self.vertices.to(device)
    
    def p2uv(self, pos, dirx, diry):
        cosTheta = torch_dot(dirx, diry)
        Inv_Dot_V1_V1 = 1/torch_dot(dirx, dirx)
        v2_ = diry - dirx * cosTheta *Inv_Dot_V1_V1 # Gram-Schmidt process

        if isinstance(self.tex, torch.Tensor):
            h, w = self.tex.shape[-2:]
        else:
            h, w = self.tex.shape[:2]
        v = (torch_dot(v2_, pos) / torch_dot(v2_, v2_))
        v.clip_(0.0, 1.0)
        u = (torch_dot(dirx, pos) *Inv_Dot_V1_V1 - cosTheta *Inv_Dot_V1_V1 * v)
        u.clip_(0.0, 1.0) 
        if self.fetchMode == 'Torch':
            uv = torch.cat([(1-v)*(h-1), ((1-u)*(w-1))], dim=1).long()
        else:
            uv = torch.cat([(1-u), (1-v)], dim=1)
        return uv

    def fetchTex(self, polygon, vec, n):
        if self.tex is None:
            return 1.0
        else:
            #* calculating the intersect point between the lookup vector and lihgting plane
            #! (tangent space, cosine distribution, un-normalized)  
            v1 = polygon[:, 1] - polygon[:, 0]
            v2 = polygon[:, 3] - polygon[:, 0]
            normal = torch_cross(v2, v1, dim=-3)
            self.setTangentLighting(polygon, normal)
            n = n.unsqueeze(1)
            #! Add a ones vector to the position where all vertices are clipped, preventing the NaN.
            vec = vec + (n == 0) * torch.ones_like(vec)
            ray = Ray(vec = vec)

            __, dist = self.intersect(ray) # the form vector and the plane must be interscted, so the boolean flag has no need to store

            p = ray.origin + ray.dir * (dist) - polygon[:, 0] # the relative position of the lighting plane
            #* convert to the uv coordinates of the texture
            uv = self.p2uv(p, v1, v2) * (~(n==0))
            #* calculating the lod (level-of-detail) of the texture according to the length of intersect ray
            area = torch_norm(normal, dim=1, keepdims=True, norm=False)
            dist = dist / area ** 0.5 / np.sqrt(2)
            lod = torch.nan_to_num(torch.log(dist*1024)/np.log(3), 0) * ~(n==0)
            
            #* interpolate the value of two adjacent lod texture
            if self.fetchMode == 'Torch':
                color = self.fetchTorch(lod, uv)
            elif self.fetchMode == 'NvDiff':
                color = self.fetchNvdiff(lod, uv)
            return color * ~(n==0)

    def fetchTorch(self, lod, uv):
        #* interpolate the value of two adjacent lod texture
        lodA = lod.floor().long().clip(0, 9)
        lodB = (lodA+1).clip(0, 9)
        if self.lod.ndim > 4:
            b,c,h,w = uv.shape
            batch = torch.arange(0, b).view(b,1,1).broadcast_to(b,h,w)
            fetchResA = self.lod[lodA[:, 0], batch, :, uv[:, 0], uv[:, 1]]
            fetchResB = self.lod[lodB[:, 0], batch, :, uv[:, 0], uv[:, 1]]
        else:
            fetchResA = self.lod[lodA[:, 0],:,uv[:, 0], uv[:, 1]]
            fetchResB = self.lod[lodB[:, 0],:,uv[:, 0], uv[:, 1]]
        color = (1- (lod-lodA)) * fetchResA.permute(0,3,1,2).contiguous() \
            + (lod-lodA) * fetchResB.permute(0,3,1,2).contiguous()
        return color
    
    def fetchNvdiff(self, lod, uv):
        if self.tex.ndim == 3:
            tex = self.tex.unsqueeze(0).permute(0,2,3,1).contiguous()
        else:
            tex = self.tex.permute(0,2,3,1).contiguous()
        uv = uv.permute(0,2,3,1).contiguous()
        lod = lod.squeeze(1)
        color = dr.texture(tex, uv, mip_level_bias=lod, mip=self.lod)
        return color.permute(0,3,1,2).contiguous()
    
class ipad():
    def __init__(self, opt):
        self.camera = Camera(opt=opt)

        self.lighting = RectLighting(opt=opt['lighting'])

    def initPosition(self):
        pass

class PolyRender(Render):
    def __init__(self, opt, lighting=None, device='cpu'):
        super().__init__(opt, lighting=lighting, device=device)
        self.lut = torch.load('deepmaterial/utils/LTC/look-up-table.pth').to(device)

    def lutToDevice(self, device):
        self.lut = self.lut.to(device)

    def render(self, svbrdf, obj_pos=None, light_pos=None, view_pos=None, keep_dirs=False, load_dirs=False, light_dir=None,
               view_dir=None, n_xy=False, r_single=True, toLDR=None, isAmbient=False):
        assert not (light_dir is not None and light_pos is not None), (
            'Given two type of lighting initilization, please choose one type (light_dir, light_pos)')
        assert not (view_dir is not None and view_pos is not None), (
            'Given two type of view initilization, please choose one type (view_dir, view_pos)')
        if not isinstance(svbrdf, PlanarSVBRDF):
            svbrdf = PlanarSVBRDF(self.opt, svbrdf)
        if toLDR is None:
            toLDR = self.opt['toLDR']
        
        n, d, r, s = svbrdf._seperate_brdf(n_xy=n_xy, r_single=r_single)
        n = n.permute(0,2,3,1).contiguous() if n.ndim == 4 else n.permute(0,1,3,4,2).contiguous()
        #* generate lighting and view direction for rendering
        if light_dir is None and light_pos is None:
            light_pos = self.lighting.vertices
        if view_pos is not None:
            if isinstance(view_pos, np.ndarray):
                view_pos = torch.from_numpy(np.array(view_pos)).view(-1, 3)
        else:
            view_pos = self.camera.fixedView()
        view_pos = view_pos.to(self.device)
        if light_dir is None:
            if not isinstance(light_pos, torch.Tensor):
                light_pos = torch.from_numpy(np.array(light_pos)).view(-1, 3)
            light_pos = light_pos.to(self.device)

            light_dir, view_dir, __, __ = self.torch_generate(view_pos, light_pos, surface=obj_pos, normLight=False)
            if keep_dirs:
                #* keep the directions in instance for next time rendering
                self.view_dir = view_dir
            if load_dirs:
                #* load the directions from instance for rendering
                view_dir = self.view_dir
        view_dir = view_dir.permute(0,2,3,1).contiguous()

        #* transform the lighting position and view direction to in local frame
        light_dir = light_dir.unsqueeze(0) # batch size
        l = Directions.worldToTangent(light_dir, n, v=view_dir)
        #* generate the lamp intensity according to the configuration
        lampIntensity = self.lighting.get_intensity(isAmbient, self.opt.get('useAug', False), self.nbRendering)
        
        if isinstance(lampIntensity, torch.Tensor):
            if len(n.shape) == 3:
                shape = [self.nbRendering, 1, 1, 1]
            if len(n.shape) == 4:
                shape = [n.shape[0], self.nbRendering, 1, 1, 1]
            lampIntensity = lampIntensity.view(*shape)

        cosTheta = torch_dot(n, view_dir, dim=-1, keepdims=False).clip(-1+1e-6, 1-1e-6)
        theta = torch.acos(cosTheta) #! be concious about the NaN (when the dot value is out of range)
        mask = ~(theta > (torch.pi/2))
        theta.nan_to_num_()
        #* fetch the ltc parameters from look-up-table
        ltc, fresNorm = self.fetchM(theta, r.squeeze(-3))
        #TODO check the nan
        if keep_dirs:
            #* keep the directions in instance for next time rendering
            self.light_dir = l
        if load_dirs:
            #* load the directions from instance for rendering
            l = self.light_dir
        #* rendering
        render_result = self.__render(d, s, l, ltc, fresNorm, intensity=lampIntensity)
        if self.nbRendering == 1:
            render_result.squeeze_(0)
        if toLDR:
            render_result = toLDR_torch(render_result)
            render_result = toHDR_torch(render_result)
        return render_result*mask

    def __render(self, d, s, l, ltc: LTC, fresNorm, useDiff=True, useSpec=True, intensity=None):
        res = 0.0
        if useSpec:
            #* specular term rendering
            s = s.squeeze(1) * 0.5 + 0.5
            f_s, lLtc, vLookUp, numLight = ltc.evalPolygon(l, s=s, fresNorm=fresNorm.squeeze(1))
            res += f_s * self.lighting.fetchTex(lLtc, vLookUp, numLight)

        if useDiff:
            #* diffuse term rendering
            d = d.squeeze(1) * 0.5 + 0.5
            f_d, lLtc, vLookUp, numLight = ltc.evalPolygon(l, isDiff=True)
            res += d * f_d * self.lighting.fetchTex(lLtc, vLookUp, numLight)

        if intensity is not None:
            res *= intensity
        return res

    def intepolateFetch(self, a, t, method='bilinear', nSampleR=64, nSampleV=64):
        if method == 'bilinear':
            #* indexing the look-up-table and fetch the parameters

            ax = torch.clip(a.floor().long(), max=nSampleR-1)
            ay = torch.clip(ax+1, max=nSampleR-1)
            tx = torch.clip(t.floor().long(), max=nSampleV-1)
            ty = torch.clip(tx+1, max=nSampleV-1)

            p1 = self.lut[ax, tx]
            p2 = self.lut[ay, tx]
            p3 = self.lut[ax, ty]
            p4 = self.lut[ay, ty]
            u = (t-tx).unsqueeze(-1)
            v = (a-ax).unsqueeze(-1)
            p = (1-u)*(1-v)*p1 + u*(1-v)*p3 + (1-u)*v*p2 + u*v*p4

        elif method=='nearest':
            a = a.round().long()
            t = t.round().long()
            p = self.lut[a,t]
        return p

    def fetchM(self, theta, r, nSampleR=64, nSampleV=64):
        #* convert the theta and roughness to the index
        t = (theta / np.pi * 2 * (nSampleV-1))
        a = ((r/2+0.5) * (nSampleR-1))

        #* indexing the look-up-table and fetch the parameters
        params = self.intepolateFetch(a, t, 'bilinear', nSampleR=nSampleR, nSampleV=nSampleV)

        #* constructing the ltc instance and store the fresnel normalization value
        ltc = LTC(m00=params[...,0], m11=params[...,1], m02=params[...,2], m20=params[...,3], amplitude=params[...,4], mode='eval')
        fresNorm = params[...,5].unsqueeze(1)
        return ltc, fresNorm

def rand_n(range_theta, range_phi, nb):
    # return normalized n (nb, 3)
    theta = np.random.uniform((range_theta[0] / 180 * np.pi), (range_theta[1] / 180 * np.pi), (nb, 1))
    phi = np.random.uniform((range_phi[0] / 180 * np.pi), (range_phi[1] / 180 * np.pi), (nb, 1))
    return np.concatenate([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1).astype(np.float32)

def random_tangent(n):
    theta_range = [0, 180]
    phi_range = [0, 360]
    theta = np.random.uniform(theta_range[0] / 180 * np.pi, theta_range[1] / 180 * np.pi)
    phi = np.random.uniform(phi_range[0] / 180 * np.pi, phi_range[1] / 180 * np.pi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    bi = np.array([x, y, z], dtype=np.float32)
    t = numpy_norm(np.cross(n, bi))

    return t

def log_normalization(img, eps=1e-2):
    return (torch.log(img + eps) - torch.log(torch.ones((1,)) * eps)) / (torch.log(1 + torch.ones((1,)) * eps) - torch.log(torch.ones((1,)) * eps))

if __name__ == "__main__":

    brdfArgs = {}
    brdfArgs['nbRendering'] = 1
    brdfArgs['size'] = 256
    brdfArgs['order'] = 'pndrs'
    brdfArgs['toLDR'] = True
    brdfArgs['lampIntensity'] = 12
    import torchvision
    if True:
        path = '/home/sda/svBRDFs/testBlended/0000001;PolishedMarbleFloor_01Xmetal_bumpy_squares;1X1.png' # [256, 256*5]indrs
        # path = '/home/sda/svBRDFs/testBlended/0000033;brick_uneven_stonesXPolishedMarbleFloor_01;2Xdefault.png'
        # path = '/home/sda/svBRDFs/testBlended/0000040;brick_uneven_stonesXleather_tiles;2X0.png'
        svbrdf = PlanarSVBRDF(brdfArgs)
        img = cv.imread(path)[:, :, ::-1] / 255.0
        svbrdf.get_svbrdfs(img) # change svbrdf.brdf
        # svbrdf.brdf[:3,:,:] = torch.stack([torch.zeros_like(svbrdf.brdf[0,:,:]),torch.zeros_like(svbrdf.brdf[0,:,:]),torch.ones_like(svbrdf.brdf[0,:,:])])
        # svbrdf.brdf[3:6,:,:] = torch.ones_like(svbrdf.brdf[0:1,:,:])*(-0.0)
        # svbrdf.brdf[6:7,:,:] = torch.ones_like(svbrdf.brdf[0:1,:,:])*(-0.8)
        # svbrdf.brdf[7:10,:,:] = torch.ones_like(svbrdf.brdf[0:1,:,:])*(1.0)
        import time
        if True:
            #====================== Rendering test============================
            renderer = Render(brdfArgs)
            start = time.time()
            res = renderer.render(svbrdf, random_light=False, colocated=True)
            print('point rendering:', time.time()-start)
            torchvision.utils.save_image(
                res**0.4545, f'tmp/test-point.png', nrow=1, padding=1, normalize=False)
        
        if True:
            #====================== Point wogamma Rendering test============================
            renderer = Render(brdfArgs)
            start = time.time()
            res = renderer.render(svbrdf, random_light=False, colocated=True)
            print('point rendering:', time.time()-start)
            torchvision.utils.save_image(
                res, f'tmp/wogammatest-point.png', nrow=1, padding=1, normalize=False)
        if True:
            #======================Parallel Rendering test============================
            renderer = Render(brdfArgs)
            start = time.time()
            res = renderer.render(svbrdf, random_light=False, light_dir = torch.tensor([0, 0.3, 1]))
            print('point rendering:', time.time()-start)
            torchvision.utils.save_image(
                res**0.4545, f'tmp/test-parallel.png', nrow=1, padding=1, normalize=False)
        if False:
            #============================ Polygonal Rendering test=======================
            # texture = cv.imread('tmp/0.png')/255
            brdfArgs['texture'] = 'tmp/0-big.png'
            brdfArgs['textureMode'] = 'Torch'
            brdfArgs['fetchMode'] = 'Torch'
            brdfArgs['nLod'] = 10
            # brdfArgs['texture'] = texture
            # rect = RectLighting(brdfArgs, [0, -1.1, 1.0], [1, 0, 0], [0, 0, 1], 1.0, 1.0) 1-4.18/2
            rect = RectLighting(brdfArgs, [1-4.18/2, 0.0, viewDistance], [1, 0, 0], [0, 1, 0], 2.91/2, 2.91/2, device='cpu')
            rect.initGaussianConv()
            renderer = PolyRender(brdfArgs, lighting=rect, device='cpu')
            # pattern = torch.load('/home/sda/klkjjhjkhjhg/DeepMaterial/experiments/Exp_0005(1)_MSAN_OptimizePattern/models/pattern_550000.pth', 'cuda').unsqueeze(0)
            # sig_pattern = (pattern - pattern.min())/(pattern.max() - pattern.min())
            # sig_pattern = torch.sigmoid(pattern)
            # rect.initTexture(sig_pattern, 'Torch')
            # svbrdf.to('cuda')
            start = time.time()
            res = renderer.render(svbrdf=svbrdf)
            # loss = res.mean()
            # rect.lod.retain_grad()
            # loss.backward(retain_graph=True)
            # torch.save(pattern.grad, 'tmp/grad2.pth')
            # for i, g in enumerate(rect.lod):
            #     torchvision.utils.save_image(g, f'tmp/fs_{i}.png')
            print('area rendering:', time.time()-start)
            print(res.min())
            torchvision.utils.save_image(
                res**0.4545, f'tmp/test-poly.png', nrow=1, padding=1, normalize=False)

    if False:
        #=============== brdf norm and importance sampling test==========================
        rect = RectLighting(brdfArgs, [0, 0, 20], [1, 0, 0], [0, 1, 0], 5, 10)
        brdf = PlanarSVBRDF(brdfArgs)
        print(brdf.norm(roughness=0.36, fresNorm=True, fresMode='UE'))
        print(brdf.avgVec(roughness=0.36))
        brdf.testSample()

    if False:
        #================================ generate the sphrical gradient illumination pictures.=================
        px = RectLighting.generateGradientLighting('x', size=(1024, 1024))
        torchvision.utils.save_image(
            px, f'tmp/x.png', nrow=1, padding=1, normalize=False)
        px = RectLighting.generateGradientLighting('y', size=(1024, 1024))
        torchvision.utils.save_image(
            px, f'tmp/y.png', nrow=1, padding=1, normalize=False)
        px = RectLighting.generateGradientLighting('z', size=(1024, 1024))
        torchvision.utils.save_image(
            px, f'tmp/z.png', nrow=1, padding=1, normalize=False)
        