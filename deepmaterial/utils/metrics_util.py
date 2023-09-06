from ast import Num
import os
from imageio import save
import numpy as np
import cv2

from deepmaterial.utils import tensor2img, imwrite, Render
from deepmaterial.utils.render_util import PlanarSVBRDF

class Metrics:
    def __init__(self,metric_type, save_imgs=True):
        self.type = metric_type
        brdf_args={
            "nbRendering": 1,
            "size": 256,
            "order": "ndrs",
            "lampIntensity": 12,
            "toLDR": False
        }
        self.save_imgs=save_imgs
        self.brdf_args = brdf_args
        self.renderer = Render(brdf_args)
        self.light_pos = [0,0,2.14]
    def __get_svbrdf_parametes(self,path, idx = 0):
        svbrdf = cv2.imread(path)/255
        svbrdf = np.split(svbrdf, 2, 0)[idx]
        if svbrdf.shape[1] == 1024:
            n, d, r, s = np.split(svbrdf,4,1)
        elif svbrdf.shape[1] == 256*5:
            p,n, d, r, s = np.split(svbrdf,5,1)
        elif svbrdf.shape[1] == 256*6:
            i, p, n, d, r, s = np.split(svbrdf, 6, 1)
        n = np.expand_dims(n,0)
        d = np.expand_dims(d,0)
        r = np.expand_dims(r,0)
        s = np.expand_dims(s,0)
        svs = np.concatenate([n,d,r,s],0)
        return svs
    def set_light_pos(self, light_pos):
        self.light_pos = light_pos
    def __get_render(self,path, num=4, idx = 0):
        svbrdf_img = cv2.imread(path)[:,:,::-1]/255
        svbrdf_img = np.split(svbrdf_img, 2, 0)[idx]
        svbrdf = PlanarSVBRDF(self.brdf_args)
        svbrdf.get_svbrdfs(svbrdf_img, num=num)
        # render_result = self.renderer.render(svbrdf, colocated=False)
        render_result = self.renderer.render(svbrdf,light_pos=self.light_pos, colocated=True)
        return tensor2img(render_result,rgb2bgr=True,out_type="float")
    def svbrdfs_from_dir(self,gt_dir,pre_dir, save_path='tmp', exp_name='exp'):
        gt_path = os.listdir(gt_dir)
        gt_path.sort()
        gt_svbrdfs = np.ones([len(gt_path),4,256,256,3])
        gt_render_results = np.ones([len(gt_path),256,256,3])
        for i,name in enumerate(gt_path):
            path = os.path.join(gt_dir,name)
            gt_svbrdfs[i] = self.__get_svbrdf_parametes(path, idx=0)
            # gt_render_results[i] = self.__get_render(path, num=5, idx = 0)
            gt_render_results[i] = self.__get_render(path, num=6, idx = 0)
        pre_path = os.listdir(pre_dir)
        pre_path.sort()
        pre_svbrdfs = np.ones([len(pre_path),4,256,256,3])
        pre_render_results = np.ones([len(pre_path),256,256,3])
        for i,name in enumerate(pre_path):
            path = os.path.join(pre_dir,name)
            pre_svbrdfs[i] = self.__get_svbrdf_parametes(path, idx=1)
            # pre_render_results[i] = self.__get_render(path, num=5, idx = 1)
            pre_render_results[i] = self.__get_render(path, num=6, idx = 1)
        if self.type == "RMSE":
            rmse_normal,rmse_diffuse,rmse_roughness,rmse_specular = self.__RMSE_svbrdf(gt_svbrdfs,pre_svbrdfs)
            render_rmse = self.__RMSE(gt_render_results,pre_render_results)
            print("RMSE for %s, re-render: %.3f, normal: %.3f, diffuse: %.3f, roughness: %.3f, specular: %.3f" % (exp_name, render_rmse,rmse_normal,rmse_diffuse,rmse_roughness,rmse_specular))
        if self.save_imgs:
            for i in range(gt_render_results.shape[0]//20):
                output_img = np.ascontiguousarray(np.concatenate([gt_render_results[i*10:(i+1)*10]**0.4545,\
                    pre_render_results[i*10:(i+1)*10]**0.4545], axis=1).transpose(1,0,2,3)).reshape(512, -1, 3)
                imwrite(output_img, os.path.join(save_path, f'renderings_{exp_name}_{i}.png'), float2int=True)


    def __RMSE_svbrdf(self,gt_svbrdf,pre_svbrdf):
        gn,gd,gr,gs = np.split(gt_svbrdf,4,1)
        pn,pd,pr,ps = np.split(pre_svbrdf,4,1)
        rmse_normal = self.__RMSE(gn,pn)
        rmse_diffuse = self.__RMSE(gd,pd)
        rmse_roughness = self.__RMSE(np.mean(gr,-1),np.mean(pr,-1))
        rmse_specular = self.__RMSE(gs,ps)
        return rmse_normal,rmse_diffuse,rmse_roughness,rmse_specular
    def __RMSE(self,gt,pre):
        mse = np.mean((gt-pre)**2)
        rmse = np.sqrt(mse)
        return rmse
    