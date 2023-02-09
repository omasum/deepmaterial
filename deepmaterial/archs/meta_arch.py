import torch
from torch import nn as nn
import math
import numpy as np

from torch.nn.functional import upsample

from deepmaterial.archs.arch_util import (MeanShift,repeat_x,RDB,Pos2Weight, Fea2Weight, ConstantWeight, AlignedFeature2Pos, input_matrix_wpn)
from torch.nn import Parameter

from deepmaterial.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class META(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch, # G
                 num_block, #self.D
                 nConv, #C
                 scale,
                 G0,
                 F0,
                 kSize,
                 img_range,
                 n_colors,
                 rgb_mean,
                 rgb_std,
                 with_edge,
                 with_lr,
                 with_videoPosMatrix,
                 ConstantWeightUpsample,
                 is_component,
                 num_frames = None,
                 name = None
                 ):
        super(META, self).__init__()
        self.scale = scale
        self.scale_idx = -1
        self.with_lr = with_lr
        self.with_edge = with_edge
        self.is_component = is_component
        self.ConstantWeightUpsample = ConstantWeightUpsample
        self.num_frames = num_frames
        # number of RDB blocks, conv layers, out channels
        self.D, C, self.G = num_block, nConv, num_out_ch
        
        if not self.is_component:
            self.sub_mean = MeanShift(img_range, rgb_mean, rgb_std)
            self.add_mean = MeanShift(img_range, rgb_mean, rgb_std, 1)

            # Shallow feature extraction net
            self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
            self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

            # Redidual dense blocks and dense feature fusion
            self.RDBs = nn.ModuleList()
            for i in range(self.D):
                self.RDBs.append(
                    RDB(growRate0=G0, growRate=self.G, nConvLayers=C)
                )
            # Global Feature Fusion
            self.GFF = nn.Sequential(*[
                nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        self.with_videoPosMatrix = with_videoPosMatrix
        ## position to weight
        self.device = 'cuda'
        if self.with_videoPosMatrix:
            self.AF2P = AlignedFeature2Pos(inC=num_in_ch, num_frame=num_frames) #in (b,t,c,h,w) out (H*W, 2*t+1)
            self.P2W = Pos2Weight(inC=num_in_ch, posLength=2*num_frames+1)
        elif self.ConstantWeightUpsample:
            self.ConsW = ConstantWeight(inC=num_in_ch)
        else:
            self.P2W = Pos2Weight(inC=num_in_ch)
        if self.with_lr or self.with_edge:
            self.F2W = Fea2Weight(inC=num_in_ch, outC=3)
   
    def input_matrix_wpn_new(self, inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = int(scale * inH), int(scale * inW)
        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH, scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)


        ####projection  coordinate  and caculate the offset
        h_project_coord = torch.arange(0., outH, 1.).mul(1.0 / scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord# Fractional part
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0., outW, 1.).mul(1.0 / scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord# Fractional part
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag, 0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)


        mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
        mask_mat = mask_mat.eq(2)

        i = 1
        h, w,_ = pos_mat.size()
        while(pos_mat[i][0][0]<= 1e-6 and i<h):
            i = i+1

        j = 1
        #pdb.set_trace()
        h, w,_ = pos_mat.size()
        while(pos_mat[0][j][1]<= 1e-6 and j<w):
            j = j+1

        pos_mat_small = pos_mat[0:i,0:j,:]

        pos_mat_small = pos_mat_small.contiguous().view(1, -1, 2)
        if add_scale:
            scale_mat = torch.zeros(1, 1)
            scale_mat[0, 0] = 1.0 / scale
            scale_mat = torch.cat([scale_mat] * (pos_mat_small.size(1)), 0)  ###(inH*inW*scale_int**2, 4)
            pos_mat_small = torch.cat((scale_mat.view(1, -1, 1), pos_mat_small), 2)

        return pos_mat_small, mask_mat  ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

        ########speed up the model by removing the computation

    def repeat_weight(self, weight, scale, inw,inh):
        k = int(math.sqrt(weight.size(0)))
        outw  =inw * scale
        outh = inh * scale
        weight = weight.view(k, k, -1)
        scale_w = (outw+k-1) // k
        scale_h = (outh + k - 1) // k
        weight = torch.cat([weight] * scale_h, 0)
        weight = torch.cat([weight] * scale_w, 1)

        weight = weight[0:outh,0:outw,:]

        return weight

    def forward(self, x, pos_mat, mask, local_weight = None):
        pos_mat = pos_mat.contiguous().view(1, -1,3)
        lr_edge = None
        if self.with_edge:
            lr_edge = x[1]
            x = x[0]
        elif self.with_lr:
            lr_in = x[1]
            x = x[0]
        elif self.with_videoPosMatrix:
            aligned_feature = x[1]
            x = x[0]
        else:
            lr_edge = None
        N,C,H,W = x.size()
        
        scale_int = math.ceil(self.scale[self.scale_idx])
        outH,outW = int(H*self.scale[self.scale_idx]),int(W*self.scale[self.scale_idx])
        # pos_mat, mask  = input_matrix_wpn(H,W, self.scale[self.scale_idx], edge = lr_edge)
        if local_weight is None:
            if self.with_videoPosMatrix:
                scale_tensor = torch.ones((1,1,1,1),dtype=torch.float32).to(self.device)*self.get_scale()
                aligned_feature = repeat_x(aligned_feature.contiguous().view(N,self.num_frames*C,H,W), upsample=True, scale_int=scale_int)
                inputs = [aligned_feature.contiguous().view(N,-1,C,aligned_feature.size(2),aligned_feature.size(3)), scale_tensor]
                pos_mat = self.AF2P(inputs)
            # pos_mat = pos_mat.to(self.device)
            if not self.is_component:
                x = self.sub_mean(x)
                f__1 = self.SFENet1(x)
                x = self.SFENet2(f__1)

                RDBs_out = []
                for i in range(self.D):
                    x = self.RDBs[i](x)
                    RDBs_out.append(x)

                x = self.GFF(torch.cat(RDBs_out, 1))
                x += f__1
            if self.ConstantWeightUpsample:
                local_weight = self.ConsW(pos_mat)
            elif self.with_edge:
                local_weight = self.F2W(pos_mat)   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
            else:
                local_weight = self.P2W(pos_mat.contiguous().view(-1, pos_mat.size(2)))
            if self.with_lr:
                lr_out = repeat_x(lr_in, upsample=True,scale_int=scale_int)
                texture_weight = self.F2W(lr_out)
                #加入边缘信息之后，一个batch中不同图片使用不同权重矩阵计算SR，所以增加一个batch_size纬度
                local_weight = local_weight.view(1, local_weight.size(0), local_weight.size(1)) * texture_weight
                del lr_out, texture_weight
                local_weight = local_weight.contiguous().view(x.size(0), x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(0,2,4,1,3,5,6).contiguous()
                local_weight = local_weight.contiguous().view(x.size(0), scale_int**2, x.size(2)*x.size(3),-1, 3)
            elif self.with_videoPosMatrix:
                local_weight = local_weight.view(N, -1, local_weight.size(1))
                local_weight = local_weight.contiguous().view(x.size(0), x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(0,2,4,1,3,5,6).contiguous()
                local_weight = local_weight.contiguous().view(x.size(0), scale_int**2, x.size(2)*x.size(3),-1, 3)
            else:
                local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
                local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)
                local_weight = local_weight.unsqueeze(0).repeat(N,1,1,1,1)

        #print(d2)
        up_x = repeat_x(x, scale_int = scale_int)     ### the output is (N*r*r,inC,inH,inW) 32

        cols = nn.functional.unfold(up_x, 3,padding=1) # 288
        # local_weight = self.repeat_weight(local_weight,scale_int,x.size(2),x.size(3))
        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3) # 870
        # del cols, local_weight
        out = out.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        if not self.is_component:
            out = self.add_mean(out)
        
        out = torch.masked_select(out,mask.to(self.device)) # 5
        out = out.contiguous().view(N,3,outH,outW)

        return out, local_weight
        
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
    def get_scale(self):
        return self.scale[self.scale_idx]