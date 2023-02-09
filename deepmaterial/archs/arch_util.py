import math
import torch
from torch import nn as nn
from torch._C import FloatStorageBase
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from deepmaterial.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from deepmaterial.utils import get_root_logger
import numpy as np
import time

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    # U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1, device='cuda'):
    y = logits + sample_gumbel(logits.size()).to(device)
    return torch.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True, dim=-1, device='cuda'):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, device)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=dim)
    y_hard = torch.zeros_like(y).view(-1, shape[-1]).to(device)
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def arg_softmax(logits, temperature=1, hard=True, dim=-1, device='cuda'):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    # y = torch.softmax(logits, dim=dim)
    y = logits

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=dim)
    y_hard = torch.zeros_like(y).view(-1, shape[-1]).to(device)
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def repeat_x(x, upsample=False, scale_int=None):
    N,C,H,W = x.size()
    x = x.view(N,C,H,1,W,1)

    x = torch.cat([x]*scale_int,3)
    x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)
    if not upsample:
        return x.contiguous().view(-1, C, H, W)
    else:
        return x.contiguous().permute(0,3,4,1,5,2).contiguous().view(N, C, H*scale_int, W*scale_int)
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False, pix_attn = False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
    
class ResidualBlockNoBNWithScaleFilter(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False, pix_attn = False):
        super(ResidualBlockNoBNWithScaleFilter, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.pix_attn = pix_attn
        # self.filter = ScaleFilter(num_feat, res_scale, pytorch_init)
        if not pix_attn:
            self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.scale_block=nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32,num_feat),
                nn.Sigmoid()
            )
        else:
        # exp31--------------
            self.scale_block=nn.Sequential(
                nn.Linear(1, 3),
                nn.ReLU(inplace=True),
                nn.Linear(3,9),
                nn.Sigmoid()
            )
        self.num_feat = num_feat
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x[0]
        y = x[1]
        x = x[0]
        if self.pix_attn:
        # exp31--------------
            weights = self.scale_block(y).contiguous().view(1, 1, 3, 3)
            self.conv1.weight.data *= weights
            out = self.conv2(self.relu(self.conv1(x)))
            out = identity + out * self.res_scale
        # exp32--------------
        else:
            weights = self.scale_block(y).contiguous().view(1, self.num_feat, 1, 1)
            self.conv3.weight.data *= weights
            out = self.relu(self.conv2(self.relu(self.conv1(x))))
            out = self.conv3(out)
            out = identity + out * self.res_scale
        # out = self.filter([out, y])
        return [out, y]

class ResidualBlockNoBNScaleAware(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---ScaleAware-|
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False, scale=None, add_scale=True, device='cuda', pix_attn=False):
        super(ResidualBlockNoBNScaleAware, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        # self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.mulConv = nn.Sequential(
            nn.Conv2d(num_feat*2, num_feat, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_feat, num_feat,3,1,1, bias=True)
        )
        self.addConv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_feat, num_feat,3,1,1, bias=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.scale_idx = -1
        self.num_feat = num_feat
        self.add_scale = add_scale
        self.device = device
        # c = 3 if add_scale else 2
        # self.attn_weights = nn.Sequential(
        #     nn.Linear(num_feat+num_feat, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32,1)
        # )
        # # self.normalization = nn.Softmax(dim=-1)
        # self.normalization = nn.Sigmoid()

        self.num_feat = num_feat
        
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        in_pos_feat = x[1]
        inverse_mask = x[2]
        x = x[0]
        identity = x
        out = self.conv2(self.relu(self.conv1(x))) # 5
        scale = self.scale[self.scale_idx]
        scale_int = math.ceil(scale)
        b,c,h,w = x.shape
        feat = torch.cat([out, in_pos_feat],dim=1) # 4
        mulFeat = self.mulConv(feat) # 5
        out = out * mulFeat + self.addConv(mulFeat) # 4
        # inverse_mask = inverse_mask.to(self.device)
        # pos_feat = self.feat_extract(inverse_pos_mat) # 36M
        # up_x = self.repeat_x(x).view(b, -1, c, h, w) # 32M
        # 32M 16M
        # attn_x = torch.cat([up_x.permute(0, 3, 4, 1, 2).contiguous(), \
        #         in_pos_feat.unsqueeze(0).repeat((b,1,1,1,1)).contiguous()],dim=-1)
        # weights = self.attn_weights(attn_x) # 80
        # weights = self.normalization(weights)*inverse_mask.view(1,h,w,-1, 1) # 16
        # eweights = torch.exp(weights)*inverse_mask.view(1,h,w,-1, 1)
        # weights = eweights/torch.sum(eweights,dim=-2, keepdim=True)

        # pos_feat = torch.sum(in_pos_feat.unsqueeze(0)*(weights),dim=-2,keepdim=False) # 1

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
        # output_weights(in_pos_feat.detach().permute(0,2,3,1).contiguous(), 'posf')
        # output_weights(mulFeat.detach().permute(0,2,3,1).contiguous(), 'mul')

        # pos_feat = torch.sum(pos_feat.unsqueeze(0)*(inverse_mask.view(1,h,w,-1, 1)),dim=-2,keepdim=False)
        # pos_feat = pos_feat/torch.sum(inverse_mask.view(1,h,w,-1),dim=-1, keepdim=True).expand(b,-1,-1,-1)
        # out = torch.cat([out,pos_feat.permute(0,3,1,2).contiguous()],dim=1) # 3
        # out = self.conv3(out) # 3
        out = identity + out * self.res_scale # -1
        # out = self.filter([out, y])
        return [out, in_pos_feat, inverse_mask]

    def set_scale(self, idx):
        self.scale_idx = idx

def build_inverse_matrix(h, w, scale_int, pos_mat, mask, add_scale=True):
    c = 3 if add_scale else 2
    pos_mat = pos_mat.contiguous().view(h,scale_int,w,scale_int,c).permute(0,2,1,3,4).contiguous().view(h,w,scale_int**2,c)
    mask = mask.contiguous().view(h,scale_int,w,scale_int).permute(0,2,1,3).contiguous().view(h,w,scale_int**2)
    return pos_mat, mask

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def input_matrix_wpn(inH, inW, scale, add_scale=True, quad=False, edge = None, reshape=True, num_gpu=1):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''
    outH, outW = int(scale * inH), int(scale * inW)

    #### mask records which pixel is invalid, 1 valid or o invalid
    #### h_offset and w_offset caculate the offset to generate the input matrix
    scale_int = int(math.ceil(scale))
    h_offset = torch.ones(inH, scale_int, 1)
    mask_h = torch.zeros(inH,  scale_int, 1)
    w_offset = torch.ones(1, inW, scale_int)
    mask_w = torch.zeros(1, inW, scale_int)
    if add_scale:
        scale_mat = torch.zeros(1,1)
        scale_mat[0,0] = 1.0/scale
        #res_scale = scale_int - scale
        #scale_mat[0,scale_int-1]=1-res_scale
        #scale_mat[0,scale_int-2]= res_scale
        scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)

    ####projection  coordinate  and caculate the offset 
    h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)#i/r
    int_h_project_coord = torch.floor(h_project_coord)#floor(i/r)

    offset_h_coord = h_project_coord - int_h_project_coord
    # offset_h_coord = h_project_coord
    int_h_project_coord = int_h_project_coord.int()#floor(i/r)

    w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)#j/r
    int_w_project_coord = torch.floor(w_project_coord)#floor(j/r)

    offset_w_coord = w_project_coord - int_w_project_coord
    # offset_w_coord = w_project_coord
    int_w_project_coord = int_w_project_coord.int()#floor(j/r)

    ####flag for   number for current coordinate LR image
    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag,  0] = 1
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
    mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
    mask_mat = mask_mat.eq(2)
    if reshape:
        pos_mat = pos_mat.contiguous().view(1, -1,2)
        if quad:
            pos_mat = torch.stack([pos_mat[:,:,0], pos_mat[:,:,1], pos_mat[:,:,0]*pos_mat[:,:,1], torch.pow(pos_mat[:,:,0], 2), torch.pow(pos_mat[:,:,1], 2)], dim=2)
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1,-1,1), pos_mat),2)
    else:
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(inH*scale_int,inW*scale_int, 1), pos_mat),2)
    if num_gpu > 1:
        pos_mat = pos_mat.view(1,1, (scale_int * inH)*(scale_int*inW), -1).repeat(num_gpu,1,1,1)
        mask_mat = mask_mat.view(1,(scale_int * inH),(scale_int*inW)).repeat(num_gpu,1,1,1)
    return pos_mat,mask_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)



class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=( kSize -1 ) //2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c* G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3, posLength = 3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(posLength,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

class ConstantWeight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(ConstantWeight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.weight = torch.Parameter(torch.Tensor(inC, outC, kernel_size, kernel_size),requires_grad=True)
        self.reset_paramenter()

    def reset_paramenter(self):
        init.kaiming_normal_(self.weight)
    def forward(self,x):
        return self.weight.view(1,-1).repeat(x.size(1),1)


class Fea2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Fea2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(self.inC, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )

    #     self.logger = SummaryWriter()
    #     self.count = 0
    # def visualize(self, feature_map):
    #     feature_map = feature_map.detach()
    #     feature_map = feature_map.reshape([-1, 1] + list(feature_map.shape)[2:4])
    #     self.logger.add_image('vis/feature_map',
    #                           torchvision.utils.make_grid(feature_map, nrow=16, normalize=True), self.count)
    #     self.count = self.count + 1

    def forward(self,x):
        N, C, H, W = x.shape
        x = x.permute(0,2,3,1).contiguous().view(-1, C)
        output = self.meta_block(x)
        return output.view(N,-1, self.kernel_size*self.kernel_size*self.inC*self.outC)

class CDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8, output_offset = False):
        super(CDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        self.offset_conv1['l1'] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2['l1'] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dcn_pack['l1'] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.feat_conv['l1'] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.output_offset = output_offset

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        offset = torch.cat([nbr_feat_l[0], ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.offset_conv1['l1'](offset))
        offset = self.lrelu(self.offset_conv2['l1'](offset))

        feat = self.dcn_pack['l1'](nbr_feat_l[0], offset)
        feat = self.feat_conv['l1'](feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        if self.output_offset:
            return feat, offset
        else:
            return feat

class TSAnPFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module with no pyramid structure.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAnPFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.pool = nn.functional.adaptive_avg_pool2d
        self.spatial_attn1 = nn.Conv2d(num_feat, num_feat, 1)
        # self.spatial_attn2 = nn.Linear(num_feat * 2, num_feat)
        self.spatial_attn2 = nn.Conv2d(num_feat, num_feat, 1, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.soft = nn.Softmax(dim=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(feat))
        # attn_max = self.max_pool(attn)
        # attn_avg = self.avg_pool(attn)
        # attn_avg = self.lrelu(nn.functional.adaptive_avg_pool2d(attn, 1))
        # attn_max = self.lrelu(nn.functional.adaptive_max_pool2d(attn, 1))
        thita = self.lrelu(self.spatial_attn2(attn)).view(b,c,-1)
        phi = self.lrelu(self.spatial_attn3(attn)).view(b,c,-1)
        g = self.lrelu(self.spatial_attn4(attn)).view(b,c,-1)

        attn = self.soft(torch.matmul(thita.permute(0,2,1), phi))
        attn = torch.matmul(attn, g.permute(0,2,1)).permute(0,2,1)

        attn = self.spatial_attn5(attn.view(b,c,h,w))
        # attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat + attn
        return feat

class AlignedFeature2Pos(nn.Module):
    def __init__(self,inC, kernel_size=3, num_frame=7):
        super(AlignedFeature2Pos,self).__init__()
        # TODO
        self.res_block = make_layer(ResidualBlockNoBN, 5, num_feat=inC*num_frame+1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_pos = nn.Conv2d(inC*num_frame+1, 2*num_frame, 3, 1, 1)
        self.conv_last = nn.Conv2d(2*num_frame, 2*num_frame, 3, 1, 1)

    def forward(self,x):
        r = x[1]
        x = x[0]
        b,t,c,h,w = x.shape
        #in (b,t,c,h*scale_int,w*scale_int) out (h*scale_int,w*scale_int, 2*t+1)
        #(b, c*t,h*scale_int,w*scale_int) (b, c*t+1,h*scale_int,w*scale_int) concat
        x = x.contiguous().view(b,t*c, h, w)
        x = torch.cat([x,r.expand(b,1,h,w)],dim=1)
        #(b, c*t+1,h*scale_int,w*scale_int) (b, c*t+1,h*scale_int,w*scale_int) conv (Res block * 5)
        x = self.res_block(x)
        #(b, c*t+1, h*scale_int, w*scale_int) (b, 2*t, h*scale_int, w*scale_int) conv
        x = self.conv_last(self.lrelu(self.conv_pos(x)))
        #(b, 2*t, h*scale_int, w*scale_int) (b, 2*t+1, h*scale_int, w*scale_int) concat
        r = torch.div(1, r)
        out = torch.cat([x,r.view(1,1,1,1).expand(b,1,h,w)],dim=1)
        out = out.contiguous().permute(0,2,3,1).view(b,-1,2*t+1)
        return out

class Offset2Pos(nn.Module):
    def __init__(self,inC, kSize=3, num_frame=7, scale=None, gradOffset=True, pLength = 64, device='cuda'):
        super(Offset2Pos,self).__init__()
        # TODO
        self.res_block = make_layer(ResidualBlockNoBN, 5, num_feat=pLength)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.conv_pos = nn.Conv2d(num_frame*pLength+1, pLength, kSize, 1, 1)
        self.conv_pos = nn.Conv2d(pLength, pLength, kSize, 1, 1)
        self.conv_last = nn.Conv2d(pLength, pLength, kSize, 1, 1)
        self.conv_upSample = nn.Conv2d(inC, pLength-1, kSize, 1, 1)
        # self.para = nn.Parameter(torch.FloatTensor(1,1,inC*kSize*kSize, 3), requires_grad=False).to(device)
        # init.kaiming_uniform_(self.para, a=math.sqrt(5))
        self.scale = scale
        self.scale_idx = -1
        self.kSize = kSize
        self.num_frame = num_frame
        self.extract = nn.functional.unfold
        self.device = device
        self.pLength = pLength
        self.gradOffset = gradOffset

    def forward(self,x, mask):
        b,t,c,h,w = x.shape
        if not self.gradOffset:
            x = x.detach()
        scale = self.scale[self.scale_idx]
        scale_int = math.ceil(self.scale[self.scale_idx])
        scale_tensor = torch.ones((1,1,1,1),dtype=torch.float32).to(self.device)*scale
        x = x.contiguous().view(b,t*c, h, w)
        # x = x[:,0]

        offset = repeat_x(x,scale_int=scale_int) # 4159
        # offset = self.extract(offset, self.kSize, padding=self.kSize//2) # 6175

        # offset = offset.contiguous().view(b*scale_int*scale_int, t*c, self.kSize, self.kSize, h, w)\
        #     .permute(0, 1, 4, 2, 5, 3).contiguous().view(scale_int*scale_int*b, t*c, self.kSize*h, self.kSize*w) # 8191
        offset_feat = self.conv_upSample(offset).view(b, scale_int, scale_int, self.pLength-1, h, w).permute(0,3,4,1,5,2).contiguous().view(b, self.pLength-1, scale_int*h, scale_int*w)
        # 7967
        # offset_feat = nn.functional.interpolate(
        #                 x, scale_factor=scale_int, mode='bicubic', align_corners=False)
        # x = offset_feat
        # x = torch.masked_select(offset_feat,mask).view(b, t*c, int(scale*h), int(scale*w))
        x = torch.masked_select(offset_feat,mask).view(b, self.pLength-1, int(scale*h), int(scale*w))
        scale_tensor = torch.ones((1,1,1,1),dtype=torch.float32).to(self.device)*(1/scale)
        x = torch.cat([x,scale_tensor.expand(b,1,int(scale*h),int(scale*w))],dim=1)
        # x = torch.cat([x,scale_tensor.expand(b,1,int(scale_int*h),int(scale_int*w))],dim=1)
        # x = self.lrelu(self.conv_pos(x))
        x = self.res_block(x)
        x = self.conv_last(x)
        out = x.contiguous()
        return out

    def set_scale(self, idx):
        self.scale_idx = idx
        
        # Copyright (c) OpenMMLab. All rights reserved.

# def flow_warp(x,
#               flow,
#               interpolation='bilinear',
#               padding_mode='zeros',
#               align_corners=True):
#     """Warp an image or a feature map with optical flow.

#     Args:
#         x (Tensor): Tensor with size (n, c, h, w).
#         flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
#             a two-channel, denoting the width and height relative offsets.
#             Note that the values are not normalized to [-1, 1].
#         interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
#             Default: 'bilinear'.
#         padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
#             Default: 'zeros'.
#         align_corners (bool): Whether align corners. Default: True.

#     Returns:
#         Tensor: Warped image or feature map.
#     """
#     if x.size()[-2:] != flow.size()[1:3]:
#         raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
#                          f'flow ({flow.size()[1:3]}) are not the same.')
#     _, _, h, w = x.size()
#     # create mesh grid
#     grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
#     grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
#     grid.requires_grad = False

#     grid_flow = grid + flow
#     # scale grid_flow to [-1,1]
#     grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
#     grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
#     grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
#     output = F.grid_sample(
#         x,
#         grid_flow,
#         mode=interpolation,
#         padding_mode=padding_mode,
#         align_corners=align_corners)
#     return output
class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x
