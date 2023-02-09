#Muti-scale Video Restoration
import torch, math
from torch import nn as nn
from deepmaterial.archs.meta_arch import META
from deepmaterial.archs.edvr_arch import PCDAlignment, TSAFusion
from deepmaterial.archs.arch_util import (repeat_x, ResidualBlockNoBN, ResidualBlockNoBNWithScaleFilter,ResidualBlockNoBNScaleAware,
                                            make_layer, input_matrix_wpn, Offset2Pos, TSAnPFusion, CDAlignment, build_inverse_matrix)
from deepmaterial.archs.basicvsrpp_arch import BasicVSRPlusPlus
from torch.nn import functional as F
from deepmaterial.utils import logger

from deepmaterial.utils.registry import ARCH_REGISTRY
import time

class UWLB(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_frames=5,
                 num_feat = 64,
                 num_f = 64,
                 num_off = 64,
                 scale = None,
                 kSize = 3,
                 with_offset=True,
                 with_f2f=False,
                 with_feature=True,
                 with_gradBack = False
                 ):
        super(UWLB, self).__init__( )
        self.scale = scale
        self.inC = num_in_ch
        self.outC = num_out_ch
        self.num_frames = num_frames
        self.num_feat = num_feat
        self.num_f = num_f
        self.num_off = num_off
        self.with_offset=with_offset
        self.with_feature=with_feature
        self.with_gradBack = with_gradBack
        self.kernel_size = kSize
        self.scale_idx = -1
        self.device = 'cuda'
        self.with_f2f = with_f2f
        if with_offset:
            self.o2p = Offset2Pos(inC=self.num_off, num_frame=num_frames, scale = self.scale, kSize = kSize, pLength=num_feat,device=self.device, gradOffset=with_gradBack)
            self.p2w = nn.Sequential(
                nn.Linear(self.num_feat+2, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256,self.kernel_size*self.kernel_size*self.num_feat*self.outC)
            )
            # self.p2w = nn.Sequential(
            #     nn.Conv2d(self.num_feat+2, 256, 1, 1),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(256,self.kernel_size*self.kernel_size*self.num_feat*self.outC, 1, 1),
            # )
        if with_feature:
            feat = num_f//4+1
            if with_f2f:
                self.f2f = Offset2Pos(inC=num_f, num_frame=num_frames, scale = self.scale, kSize = kSize, pLength=num_feat,device=self.device, gradOffset=with_gradBack)
                feat = self.num_feat
            else:
                self.feat_ext = nn.Sequential(
                    nn.Conv2d(num_f, num_f//4, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_f//4, num_f//4, 3, 1, 1),
                )
            self.f2w = nn.Sequential(
                nn.Linear(feat, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256,self.kernel_size*self.kernel_size*self.num_feat*self.outC)
            )
            # self.f2w = nn.Sequential(
            #     nn.Conv2d(feat, 256, 1, 1),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(256,self.kernel_size*self.kernel_size*self.num_feat*self.outC, 1, 1),
            # )
        self.extract = nn.functional.unfold

    def expand_weights(self, w, mask):
        
        m = mask.view(-1)
    def forward(self, x, pos_mat, mask, weights=None):
        b, t, c, h, w = x[1].shape # 2819
        scale_int = math.ceil(self.scale[self.scale_idx])
        outH, outW = int(self.scale[self.scale_idx]*h), int(self.scale[self.scale_idx]*w)
        H, W = int(scale_int*h), int(scale_int*w)
        # offset = self.repeat_x(x[2].view(b,-1,h,w),upsample=True).contiguous().view(b, t*c, -1)
        offset = x[2]
        feature = x[1]# 3655
        if weights is None:
            if self.with_offset:
                offset_mat = self.o2p(offset, mask)# 5133
                place_holder = torch.ones(b,self.num_feat,H,W,dtype=torch.float32,device=self.device)
                # place_holder = torch.FloatTensor(b,self.num_feat,H,W).to(self.device)
                offset_mat = place_holder.masked_scatter(mask, offset_mat)
                offset_mat = torch.cat([offset_mat, pos_mat.unsqueeze(0).permute(0,3,1,2).contiguous().expand(b,-1,-1,-1)[:,1:,:,:]], dim=1)# 5969
                offset_mat = offset_mat.contiguous().view(b,-1,H*W).permute(0,2,1).contiguous()
                offset_weights = self.p2w(offset_mat)# 5969
                # tmp = offset_weights.view(b, H, W, -1)
                # tmp = tmp.cpu().numpy()
            if self.with_feature:
                if not self.with_gradBack:
                    feature = feature.detach()
                if self.with_f2f:
                    feat_mat = self.f2f(feature, mask)
                    place_holder = torch.ones(b,self.num_feat,H,W,dtype=torch.float32,device=self.device)
                # place_holder = torch.FloatTensor(b,self.num_feat,H,W).to(self.device)
                    place_holder = place_holder.masked_scatter(mask, feat_mat)
                    feat_mat = place_holder.contiguous().view(b,self.num_feat,H*W).permute(0,2,1).contiguous()
                else:
                    feature_up = repeat_x(feature.view(b,-1,h,w),upsample=True, scale_int = scale_int)# 5805
                    feat_mat = self.feat_ext(feature_up)# 7599
                    feat_mat = torch.cat([feat_mat, pos_mat.unsqueeze(0).permute(0,3,1,2).contiguous().expand(b,-1,-1,-1)[:,0:1,:,:]], dim=1)
                    feat_mat = feat_mat.contiguous().view(b, t*(c//4)+1, -1)
                    feat_mat = feat_mat.contiguous().permute(0,2,1).contiguous() # 6641
                    # feat_mat = torch.cat([feat_mat, pos_mat.unsqueeze(0)[:,:,:,0:1].expand(b,-1,-1,-1).view(b,-1,1)], dim=-1) # 6867
                feat_weights = self.f2w(feat_mat) # 8243
            if self.with_feature and self.with_offset:
                weights = offset_weights + feat_weights # 8463
                # weights = weights.permute(0, 2,3,1).contiguous()
                # weights = offset_weights
            elif not self.with_feature:
                weights = offset_weights 
            elif not self.with_offset:
                weights = feat_weights
            
            # def output_weights(weights, name):
            #     import numpy as np
            #     import cv2
            #     b, outH, outW, _ = weights.shape
            #     sumall = torch.zeros((b,outH, outW), dtype=torch.float32).to('cuda')
            #     tmp = weights.view(b, outH, outW, -1)
            #     for i in range(tmp.shape[-1]):
            #         c = tmp[0,:,:, i]
            #         sumall+=c
            #         # c = (c-torch.min(c))/(torch.max(c)-torch.min(c))
            #         # c = c.cpu().numpy()
            #         # cv2.imwrite('tmp/'+name+'_{%3d}.png'%i, (c*255).astype(np.uint8))
            #     sumall = sumall/tmp.shape[-1]
            #     sumall = (sumall-torch.min(sumall))/(torch.max(sumall)-torch.min(sumall))
            #     sumall = sumall.cpu().numpy()
            #     cv2.imwrite('tmp/'+name+'.png', (sumall[0]*255).astype(np.uint8))
            # weights = offset_weights.contiguous().view(b, h,scale_int, w,scale_int,-1,self.outC).permute(0, 5,6, 1,2,3,4).contiguous()
            # weights = torch.masked_select(weights.view(b, -1, H, W),mask.to(self.device))
            # weights = weights.contiguous().view(b,-1,outH,outW).permute(0,2,3,1).contiguous()
            # weights = weights.detach()
            # output_weights(weights, 'offset')
            # output_weights(pos_mat.unsqueeze(0), 'pos')
            # output_weights(feature.detach().view(b, -1, h, w).permute(0,2,3,1).contiguous(), 'f')
            # output_weights(offset.detach().view(b, -1, h, w).permute(0,2,3,1).contiguous(), 'o')
            # weights = feat_mat.contiguous().view(b, h,scale_int, w,scale_int,-1).permute(0, 5, 1,2,3,4).contiguous()
            # weights = weights.detach()
            # output_weights(weights.view(b, -1, H, W).permute(0,2,3,1).contiguous(), 'img')
            # weights = feat_weights.contiguous().view(b, h,scale_int, w,scale_int,-1).permute(0, 5, 1,2,3,4).contiguous()
            # weights = torch.masked_select(weights.view(b, -1, H, W),mask.to(self.device))
            # weights = weights.contiguous().view(b,-1,outH,outW).permute(0,2,3,1).contiguous()
            # weights = weights.detach()
            # output_weights(weights, 'feat')

            mid = time.time() # 9107
            weights = weights.contiguous().view(b, h,scale_int, w,scale_int,-1,self.outC).permute(0, 2,4,1,3,5,6).contiguous() # 9523
            weights = weights.contiguous().view(b, scale_int**2, h*w,-1, self.outC)
        x = x[0]
        ## print(d2)
        up_x = repeat_x(x, scale_int=scale_int)     ### the output is (N*r*r,inC,inH,inW) # 32

        cols = nn.functional.unfold(up_x, 3,padding=1) # 288
        # local_weight = self.repeat_weight(local_weight,scale_int,x.size(2),x.size(3))
        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        out = torch.matmul(cols,weights).permute(0,1,4,2,3) # 6
        out = out.contiguous().view(x.size(0),scale_int,scale_int,self.outC,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),self.outC, scale_int*x.size(2),scale_int*x.size(3))
        
        out = torch.masked_select(out,mask.to(self.device))
        out = out.contiguous().view(b,self.outC,outH,outW)

        return out, weights

    def set_scale(self, idx):
        self.scale_idx = idx
        if self.with_offset:
            self.o2p.set_scale(idx)
        if self.with_feature and self.with_f2f:
            self.f2f.set_scale(idx)
    
    def get_scale(self):
        return self.scale[self.scale_idx]


@ARCH_REGISTRY.register()
class MSVR(nn.Module):
    def __init__(self,
                 name = None,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 kSize=3,
                 num_edge_feat=16,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 hr_in=False,
                 is_bicubic=False,
                 vis_feature = False,
                 pix_attn = False,
                 with_basicvsrpp=False,
                 with_pcd=True,
                 with_cd=False,
                 with_recons=True,
                 with_scaleAware=False,
                 with_tsa=False,
                 with_tsanp=False,
                 with_meta=False,
                 with_edge=False,
                 with_lr = False,
                 with_scalefilter = False,
                 with_videoPosMatrix = False,
                 with_resMeta = False,
                 with_resNfMeta = False,
                 with_onlyMeta = False,
                 with_resFeat = False,
                 with_uwlb = False,
                 meta_args=None,
                 uwlb_args=None,
                 vpp_args = None):
        super(MSVR, self).__init__()

        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.is_bicubic = is_bicubic
        self.vis_feature = vis_feature
        self.with_pcd = with_pcd
        self.with_cd = with_cd
        self.with_tsa = with_tsa
        self.with_tsanp = with_tsanp
        self.with_recons = with_recons
        self.with_meta = with_meta
        self.with_edge = with_edge
        self.with_lr = with_lr
        self.with_scalefilter = with_scalefilter
        self.with_videoPosMatrix = with_videoPosMatrix
        self.with_resMeta = with_resMeta
        self.with_onlyMeta = with_onlyMeta
        self.with_resNfMeta = with_resNfMeta
        self.with_resFeat = with_resFeat
        self.with_uwlb = with_uwlb
        self.with_scaleAware = with_scaleAware
        self.device = 'cuda'
        self.with_basicvsrpp = with_basicvsrpp
        self.is_component_vpp = vpp_args['is_component']
        self.num_feat = num_feat
        meta_args['F0'] = num_feat
        meta_args['G0'] = num_feat
        self.scale_log={}
        # extract features for each frame
        if self.with_onlyMeta:
            self.upsample_meta_res = META(num_in_ch, 3, is_component=True, with_edge=False, with_lr=False, with_videoPosMatrix=False, num_frames=num_frame, kSize=kSize, **meta_args)
            return
        if self.with_basicvsrpp and self.is_component_vpp:
            self.feature_learning = BasicVSRPlusPlus(num_feat, with_scaleAware=self.with_scaleAware, **vpp_args)
            if self.with_scaleAware:
                c = 3
                self.posf_extract = nn.Sequential(
                    nn.Linear(c, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64,self.num_feat)
                )
                self.attn = nn.Sequential(
                    nn.Linear(num_feat+3, num_feat),
                    nn.ReLU(inplace=True),
                    nn.Linear(num_feat,num_feat)
                )
                self.sig = nn.Sigmoid()
        elif self.with_basicvsrpp:
            self.generator = BasicVSRPlusPlus(num_feat, with_scaleAware=self.with_scaleAware, **vpp_args)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            # extrat pyramid features
            self.feature_extraction = make_layer(
                ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
            
            # pcd and tsa module
            if self.with_pcd:
                self.pcd_align = PCDAlignment(
                    num_feat=num_feat, deformable_groups=deformable_groups, output_offset=self.with_uwlb)
                self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
                self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
                self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            elif self.with_cd:
                self.pcd_align = CDAlignment(
                    num_feat=num_feat, deformable_groups=deformable_groups, output_offset=self.with_uwlb)
            else:
                self.align = nn.Conv2d(num_frame * num_feat, num_frame * num_feat, 1, 1)
            
            if self.with_tsa:
                self.fusion = TSAFusion(
                    num_feat=num_feat,
                    num_frame=num_frame,
                    center_frame_idx=self.center_frame_idx)
            elif self.with_tsanp:
                self.fusion = TSAnPFusion(
                    num_feat=num_feat,
                    num_frame=num_frame,
                    center_frame_idx=self.center_frame_idx)
            else:
                self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

            if self.with_recons:
                # reconstruction
                if self.with_scalefilter:
                    resblock = ResidualBlockNoBNWithScaleFilter
                    self.reconstruction = make_layer(
                        resblock, num_reconstruct_block, num_feat=num_feat, pix_attn=pix_attn)
                elif self.with_scaleAware:
                    resblock = ResidualBlockNoBNScaleAware
                    self.reconstruction = make_layer(
                        resblock, num_reconstruct_block, num_feat=num_feat, scale = meta_args['scale'], device=self.device)
                else:
                    resblock = ResidualBlockNoBN
                    self.reconstruction = make_layer(
                        resblock, num_reconstruct_block, num_feat=num_feat, pix_attn=pix_attn)
        
        self.meta_args = meta_args
        # upsample
        if self.with_lr:
            inC = 3
            self.edge_feature_extraction = nn.Conv2d(inC, num_edge_feat, 3, 1, 1)
            self.edge_fusion = nn.Conv2d(num_frame * num_edge_feat, num_edge_feat, 3, 1, 1)
        if self.with_meta:
            #meta_args:num_block, nConv, 
            # self.upsample_meta = META(num_feat,num_out_ch, is_component=True, with_edge=self.with_edge, with_lr=self.with_lr, with_videoPosMatrix=self.with_videoPosMatrix, num_frames=num_frame, kSize=kSize, **meta_args)
            self.upsample_meta = META(num_feat, 3, is_component=True, with_edge=False, with_lr=False, with_videoPosMatrix=False, num_frames=num_frame, kSize=kSize, **meta_args)
        elif self.with_uwlb:
            if self.with_basicvsrpp:
                num_off = 4*(num_frame-1)
                num_f = num_feat*num_frame
                self.fuse_aligned_feat = nn.Sequential(
                    nn.Conv2d(4*num_f, num_feat*num_frame, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_feat*num_frame,num_feat*num_frame, 1, 1),
                )
            else:
                num_off = num_feat*num_frame
                num_f = num_feat*num_frame
            self.UWLB = UWLB(num_in_ch, 3, num_frames=num_frame, num_feat = num_feat, num_f=num_f, num_off= num_off, kSize=kSize, **uwlb_args)
        elif self.with_basicvsrpp:
            pass
        else:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            
        if self.with_resMeta or self.with_resNfMeta:
            self.upsample_meta_res = META(num_in_ch, 3, is_component=True, with_edge=False, with_lr=False, with_videoPosMatrix=False, num_frames=num_frame, kSize=kSize, **meta_args)
            if self.with_resNfMeta:
                self.avg_pool_hr = torch.mean
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    
    def upsample_vpp(self, lqs, feats, upsample_net, res_net=None, pos_mat=None, mask=None, inv_mat=None, inv_mask=None, aligned_feat=None, offsets=None):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]
        weights = None
        weights_res = None
        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            # if self.cpu_cache:
            #     hr = hr.cuda()

            if self.with_scaleAware:
                hr = self.feature_learning.reconstruction(hr, inv_mat[:,i], inv_mask)
            else:
                hr = self.feature_learning.reconstruction(hr, inv_mat, inv_mask)
            if self.with_scaleAware:
                hr = hr[0]
            if self.with_uwlb:
                hr = [hr, aligned_feat, offsets]
                hr, weights = upsample_net(hr, pos_mat, mask, weights)
            elif self.with_meta:
                hr, weights = upsample_net(hr, pos_mat, mask, weights)
            if res_net is not None:
                up_hr, weights_res = res_net(lqs[:, i, :, :, :], pos_mat, mask, weights_res)
                hr += up_hr
            # if self.cpu_cache:
            #     hr = hr.cpu()
            #     torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        if self.with_edge:
            edge = x[1]
            x = x[0]

        begin = time.time()
        b, t, c, h, w = x.size()
        # if self.hr_in:
        #     assert h % 16 == 0 and w % 16 == 0, (
        #         'The height and width must be multiple of 16.')
        # else:
        #     assert h % 4 == 0 and w % 4 == 0, (
        #         'The height and width must be multiple of 4.')
        inverse_pos_mat, inverse_mask = None, None
        pos_mat, mask = None, None
        start = time.time()
        if self.with_onlyMeta or self.with_meta or self.with_resMeta or self.with_scaleAware or self.with_uwlb:
            scale = self.get_scale()
            scale_int = math.ceil(scale)
            pos_mat, mask = input_matrix_wpn(h,w,scale, reshape=False)
            if self.with_scaleAware:
                inverse_pos_mat, inverse_mask = build_inverse_matrix(h,w,scale_int, pos_mat, mask)
                inverse_pos_mat = inverse_pos_mat.to(self.device)
                inverse_mask = inverse_mask.to(self.device)
                if self.with_basicvsrpp:
                    pos_feat = self.posf_extract(inverse_pos_mat).unsqueeze(0).unsqueeze(0).repeat((b,t,1,1,1,1)) # 240
                    up_x = repeat_x(x.view(-1, c, h, w), scale_int = scale_int).view(b, t, -1, c, h, w) # 42
                    attn_x = torch.cat([up_x.permute(0, 1, 4, 5, 2, 3).contiguous(), \
                            pos_feat.contiguous()],dim=-1) # 266
                    
                    weights = self.attn(attn_x) # 448
                    # eweights = torch.exp(weights)*inverse_mask.view(1,h,w,-1, 1) # 28
                    # weights = eweights/torch.sum(eweights,dim=-2, keepdim=True) # 1
                    weights = self.sig(weights*inverse_mask.view(1,1,h,w,-1, 1))

                    pos_feat = torch.sum(pos_feat*(weights),dim=-2,keepdim=False) # 14
                    inverse_pos_mat = pos_feat.permute(0,1,4,2,3).contiguous()
                # inverse_pos_mat = torch.sum(inverse_pos_mat,dim=2,keepdim=False)/torch.sum(inverse_mask,dim=2, keepdim=True)
                # if scale_int == 4:
                #     max_pool = self.max_pool_4
                # elif scale_int == 3:
                #     max_pool = self.max_pool_3
                # else:
                #     max_pool = self.max_pool_2
                # inverse_pos_mat = inverse_pos_mat.permute(0,4,3,1,2).contiguous()
                # inverse_pos_mat = max_pool(inverse_pos_mat).squeeze(2).repeat(b,1,1,1)
                # inverse_pos_mat = inverse_pos_mat.permute(0).contiguous().squeeze(2).repeat(b,1,1,1)
                
            pos_mat = pos_mat.to(self.device)
            mask = mask.to(self.device)
            # print("prepare time:",time.time()-start)

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()
        if self.with_onlyMeta:
            hr = self.upsample_meta_res(x_center, pos_mat, mask)
            return hr
        if self.with_basicvsrpp:
            inputs = [x, inverse_pos_mat, inverse_mask]
            if self.is_component_vpp:
                start = time.time()
                out, aligned_feat, offsets = self.feature_learning(inputs)
                # print("main network time:",time.time()-start)
            else:
                out = self.generator(inputs)
            if self.with_uwlb:
                upsample_net = self.UWLB
                offsets = torch.cat(offsets, dim=2)
                aligned_feat = torch.cat([torch.stack(aligned_feat[f],dim=1) for f in aligned_feat], dim=2).view(b,-1,h,w)
                aligned_feat = self.fuse_aligned_feat(aligned_feat).view(b, t, self.num_feat, h, w)
            elif self.with_meta:
                upsample_net = self.upsample_meta
            else:
                upsample_net = None
            if self.with_resMeta:
                res = self.upsample_meta_res
            else:
                res = None
            if upsample_net is not None:
                start = time.time()
                out = self.upsample_vpp(x, out, upsample_net, res, pos_mat=pos_mat, mask=mask, inv_mat=inverse_pos_mat, inv_mask=inverse_mask, offsets= offsets, aligned_feat=aligned_feat)
                # print("upsample time:",time.time()-start)
            # else:
            #     out = self.generator.upsample(x, out)
        else:
            # extract features for each frame
            # L1
            # out = x_center
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
            feat_res = feat_l1.view(b, t, -1, h, w)

            if self.with_pcd:
                feat_l1 = self.feature_extraction(feat_l1)
                # L2
                feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
                feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
                # L3
                feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
                feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

                feat_l1 = feat_l1.view(b, t, -1, h, w)
                feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
                feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

                # PCD alignment
                ref_feat_l = [  # reference feature list
                    feat_l1[:, self.center_frame_idx, :, :, :].clone(),
                    feat_l2[:, self.center_frame_idx, :, :, :].clone(),
                    feat_l3[:, self.center_frame_idx, :, :, :].clone()
                ]
                aligned_feat = []
                offsets = []
                for i in range(t):
                    nbr_feat_l = [  # neighboring feature list
                        feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                        feat_l3[:, i, :, :, :].clone()
                    ]
                    if self.with_uwlb:
                        feature, offset = self.pcd_align(nbr_feat_l, ref_feat_l)
                        offsets.append(offset)
                    else:
                        feature = self.pcd_align(nbr_feat_l, ref_feat_l)
                    aligned_feat.append(feature)
                if self.with_uwlb:
                    offsets = torch.stack(offsets, dim=1)  # (b, t, c, h, w)
                aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)
            elif self.with_cd:
                feat_l1 = self.feature_extraction(feat_l1)
                feat_l1 = feat_l1.view(b, t, -1, h, w)

                # PCD alignment
                ref_feat_l = [  # reference feature list
                    feat_l1[:, self.center_frame_idx, :, :, :].clone()
                ]
                aligned_feat = []
                offsets = []
                for i in range(t):
                    nbr_feat_l = [  # neighboring feature list
                        feat_l1[:, i, :, :, :].clone()
                    ]
                    if self.with_uwlb:
                        feature, offset = self.pcd_align(nbr_feat_l, ref_feat_l)
                        offsets.append(offset)
                    else:
                        feature = self.pcd_align(nbr_feat_l, ref_feat_l)
                    aligned_feat.append(feature)
                if self.with_uwlb:
                    offsets = torch.stack(offsets, dim=1)  # (b, t, c, h, w)
                aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)
                feat_l2 = None
            else:
                aligned_feat = self.align(feat_l1.view(b,t, -1 ,h,w).view(b,-1,h,w))
                aligned_feat = aligned_feat.view(b, t, -1, h, w)
                feat_l2 = None
            if not self.with_tsa and not self.with_tsanp:
                aligned_feat = aligned_feat.view(b, -1, h, w)
            out = self.fusion(aligned_feat)
            
            if self.with_recons:
                if self.with_scalefilter:
                    scale_tensor = torch.ones((1,1),dtype=torch.float32).to(self.device)*self.get_scale()
                    inputs = [out, scale_tensor]
                else:
                    inputs = out
            
                if self.with_scaleAware:
                    inputs = [out, inverse_pos_mat, inverse_mask]
                out = self.reconstruction(inputs)
            
                if self.with_scalefilter or self.with_scaleAware:
                    out = out[0]
            if self.vis_feature:
                before_up = out
            if self.with_meta:
                if self.with_resFeat:
                    out += feat_res[:,self.center_frame_idx,...]
                if self.with_videoPosMatrix:
                    out = [out, aligned_feat]
                if self.with_edge or self.with_lr:
                    if self.with_edge:
                        out = [out, edge[:,self.center_frame_idx,:,:]]
                    elif self.with_lr:
                        edge = x.view(-1,3,h,w) 
                        feat_edge = self.edge_feature_extraction(edge)
                        feat_edge = feat_edge.view(b,-1,h,w)
                        fused_edge = self.edge_fusion(feat_edge)
                        out = [out, fused_edge]
                out, _ = self.upsample_meta(out, pos_mat, mask)
            elif self.with_uwlb:
                inputs = [out, aligned_feat, offsets]
                out, _ = self.UWLB(inputs, pos_mat, mask)
            else:
                out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
                out = self.lrelu(self.conv_hr(out))
                out = self.conv_last(out)
            if self.with_resMeta:
                hr, _ = self.upsample_meta_res(x_center, pos_mat, mask)
                out += hr
                # out = hr
                # pass
            elif self.with_resNfMeta:
                hr = []
                for i in range(t):
                    base = x[:, i, :, :, :].contiguous()
                    hr.append(self.upsample_meta_res(base, pos_mat, mask))
                hr = torch.stack(hr,dim=1)
                hr = self.avg_pool_hr(hr, dim=1)
                out += hr
            else:
                if self.hr_in:
                    base = x_center
                else:
                    base = F.interpolate(
                        x_center, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=False)
                out += base
                # out = base
            if self.vis_feature:
                return before_up
        # print("inner forward time:",time.time()-begin)
        return out

    def set_scale(self, scale_idx):
        if self.with_meta:
            self.upsample_meta.set_scale(scale_idx)
        elif self.with_uwlb:
            self.UWLB.set_scale(scale_idx)
        if not self.with_basicvsrpp and self.with_scaleAware:
            for i in range(len(self.reconstruction)):
                self.reconstruction[i].set_scale(scale_idx)
        elif self.with_basicvsrpp:
            if self.is_component_vpp:
                self.feature_learning.set_scale(scale_idx)
        if self.with_resMeta or self.with_resNfMeta or self.with_onlyMeta:
            self.upsample_meta_res.set_scale(scale_idx)
    def get_scale(self):
        if self.with_meta:
            return self.upsample_meta.get_scale()
        elif self.with_uwlb:
            return self.UWLB.get_scale()
        elif self.with_resMeta or self.with_resNfMeta or self.with_onlyMeta:
            return self.upsample_meta_res.get_scale()
        else:
            return 4