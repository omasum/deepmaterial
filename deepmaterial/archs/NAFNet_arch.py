# ------------------------------------------------------------------------
# Copyright (c) 2022 Murufeng. All Rights Reserved.
# ------------------------------------------------------------------------
'''
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepmaterial.utils.materialmodifier import materialmodifier_L6
from deepmaterial.utils.registry import ARCH_REGISTRY

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


@ARCH_REGISTRY.register()
class NAFNet(nn.Module):

    def __init__(self, in_channel=3, out_channel=10, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], res=False, tanh=True, **kwargs):
        super().__init__()

        self.res = res
        self.tanh = nn.Tanh() if tanh else tanh
        self.usePattern = kwargs.get('usePattern', False)
        if self.usePattern:
            self.intro = nn.Conv2d(in_channels=in_channel, out_channels=int(width*3/4), kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
            self.introPattern = self.build_intropattern(int(width/4))
        else:
            self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.encoders, self.middle_blks, self.downs, width = self.build_encoders(width, enc_blk_nums, middle_blk_num)
        self.ups, self.decoders, self.padder_size, width = self.build_decoders(width, dec_blk_nums)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

    def build_intropattern(self, width):
        introPattern = []
        introPattern.append(nn.Conv2d(in_channels=3, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width))
        introPattern.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width))
        return nn.Sequential(*introPattern)

    def build_encoders(self, width, enc_blk_nums, middle_blk_num):
        encoders = nn.ModuleList()
        downs = nn.ModuleList()
        for num in enc_blk_nums:
            encoders.append(
                nn.Sequential(
                    *[NAFBlock(width) for _ in range(num)]
                )
            )
            downs.append(
                nn.Conv2d(width, 2*width, 2, 2)
            )
            width = width * 2

        middle_blks = \
            nn.Sequential(
                *[NAFBlock(width) for _ in range(middle_blk_num)]
            )
        return encoders, middle_blks, downs, width

    def build_decoders(self, width, dec_blk_nums):
        ups = nn.ModuleList()
        decoders = nn.ModuleList()
        for num in dec_blk_nums:
            ups.append(
                nn.Sequential(
                    nn.Conv2d(width, width * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            width = width // 2
            decoders.append(
                nn.Sequential(
                    *[NAFBlock(width) for _ in range(num)]
                )
            )

        padder_size = 2 ** len(self.encoders)
        return ups, decoders, padder_size, width

    def forward(self, inp, pattern=None):
        B, C, H, W = inp.shape
        self.HighFrequency = torch.ones(B, C, int(H/2), int(W/2))
        inp = self.check_image_size(inp)

        x = self.intro(inp) # [b,width,h,w]
        if self.usePattern:
            featPattern = self.introPattern(pattern)
            x = torch.cat([x, featPattern.broadcast_to(B,featPattern.shape[1], H, W)], dim = 1)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x) # [8,512,16,16]

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        if self.res:
            x = x + inp
        if self.tanh:
            x = self.tanh(x)
        else:
            x = x.clamp(-1+1e-6, 1-1e-6)
        return self.HighFrequency, x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
@ARCH_REGISTRY.register()
class NAFSDNet(nn.Module):

    def __init__(self, in_channel=5, out_channel=10, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], res=False, tanh=True, **kwargs):
        super().__init__()

        self.res = res
        self.tanh = nn.Tanh() if tanh else tanh
        self.usePattern = kwargs.get('usePattern', False)
        if self.usePattern:
            self.intro = nn.Conv2d(in_channels=in_channel, out_channels=int(width*3/4), kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
            self.introPattern = self.build_intropattern(int(width/4))
        else:
            self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.encoders, self.middle_blks, self.downs, width = self.build_encoders(width, enc_blk_nums, middle_blk_num)
        self.nups, self.ndecoders, self.padder_size, widthn = self.build_decoders(width, dec_blk_nums)
        self.dups, self.ddecoders, self.dpadder_size, widthd = self.build_decoders(width, dec_blk_nums)
        self.rups, self.rdecoders, self.rpadder_size, widthr = self.build_decoders(width, dec_blk_nums)
        self.sups, self.sdecoders, self.spadder_size, widths = self.build_decoders(width, dec_blk_nums)
        self.nending = nn.Conv2d(in_channels=widthn, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.dending = nn.Conv2d(in_channels=widthd, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.rending = nn.Conv2d(in_channels=widthr, out_channels=1, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.sending = nn.Conv2d(in_channels=widths, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

    def build_intropattern(self, width):
        introPattern = []
        introPattern.append(nn.Conv2d(in_channels=3, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width))
        introPattern.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width))
        return nn.Sequential(*introPattern)

    def build_encoders(self, width, enc_blk_nums, middle_blk_num):
        encoders = nn.ModuleList()
        downs = nn.ModuleList()
        for num in enc_blk_nums:
            encoders.append(
                nn.Sequential(
                    *[NAFBlock(width) for _ in range(num)]
                )
            )
            downs.append(
                nn.Conv2d(width, 2*width, 2, 2)
            )
            width = width * 2

        middle_blks = \
            nn.Sequential(
                *[NAFBlock(width) for _ in range(middle_blk_num)]
            )
        return encoders, middle_blks, downs, width

    def build_decoders(self, width, dec_blk_nums):
        ups = nn.ModuleList()
        decoders = nn.ModuleList()
        for num in dec_blk_nums:
            ups.append(
                nn.Sequential(
                    nn.Conv2d(width, width * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            width = width // 2
            decoders.append(
                nn.Sequential(
                    *[NAFBlock(width) for _ in range(num)]
                )
            )

        padder_size = 2 ** len(self.encoders)
        return ups, decoders, padder_size, width

    def forward(self, inp, pattern=None):
        B, C, H, W = inp.shape
        self.HighFrequency = torch.ones(B, C, int(H/2), int(W/2))
        inp = self.check_image_size(inp)

        x = self.intro(inp) # [b,width,h,w]
        if self.usePattern:
            featPattern = self.introPattern(pattern)
            x = torch.cat([x, featPattern.broadcast_to(B,featPattern.shape[1], H, W)], dim = 1)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x) # [8,512,16,16]
        nx = x
        dx = x
        rx = x
        sx = x

        for decoder, up, enc_skip in zip(self.ndecoders, self.nups, encs[::-1]):
            nx = up(nx)
            nx = nx + enc_skip
            nx = decoder(nx)

        nx = self.nending(nx)
        for decoder, up, enc_skip in zip(self.ddecoders, self.dups, encs[::-1]):
            dx = up(dx)
            dx = dx + enc_skip
            dx = decoder(dx)

        dx = self.dending(dx)
        for decoder, up, enc_skip in zip(self.rdecoders, self.rups, encs[::-1]):
            rx = up(rx)
            rx = rx + enc_skip
            rx = decoder(rx)

        rx = self.rending(rx)
        for decoder, up, enc_skip in zip(self.sdecoders, self.sups, encs[::-1]):
            sx = up(sx)
            sx = sx + enc_skip
            sx = decoder(sx)

        sx = self.sending(sx)
        if self.res:
            nx = nx + inp
            dx = dx + inp
            rx = rx + inp
            sx = sx + inp
        if self.tanh:
            nx = self.tanh(nx)
            dx = self.tanh(dx)
            rx = self.tanh(rx)
            sx = self.tanh(sx)
        else:
            x = x.clamp(-1+1e-6, 1-1e-6)
        output = torch.cat([nx, dx, rx, sx], dim=1)
        return self.HighFrequency, output

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
@ARCH_REGISTRY.register()
class NAFNSDNet(nn.Module):

    def __init__(self, in_channel=5, out_channel=10, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], res=False, tanh=True, **kwargs):
        super().__init__()

        self.res = res
        self.tanh = nn.Tanh() if tanh else tanh
        self.usePattern = kwargs.get('usePattern', False)
        if self.usePattern:
            self.intro = nn.Conv2d(in_channels=in_channel, out_channels=int(width*3/4), kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
            self.introPattern = self.build_intropattern(int(width/4))
        else:
            self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.encoders, self.middle_blks, self.downs, width = self.build_encoders(width, enc_blk_nums, middle_blk_num)
        self.nups, self.ndecoders, self.padder_size, widthn = self.build_decoders(width, dec_blk_nums)
        self.ups, self.decoders, self.padder_size, width = self.build_decoders(width, dec_blk_nums)
        self.nending = nn.Conv2d(in_channels=widthn, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=7, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

    def build_intropattern(self, width):
        introPattern = []
        introPattern.append(nn.Conv2d(in_channels=3, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width))
        introPattern.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width))
        return nn.Sequential(*introPattern)

    def build_encoders(self, width, enc_blk_nums, middle_blk_num):
        encoders = nn.ModuleList()
        downs = nn.ModuleList()
        for num in enc_blk_nums:
            encoders.append(
                nn.Sequential(
                    *[NAFBlock(width) for _ in range(num)]
                )
            )
            downs.append(
                nn.Conv2d(width, 2*width, 2, 2)
            )
            width = width * 2

        middle_blks = \
            nn.Sequential(
                *[NAFBlock(width) for _ in range(middle_blk_num)]
            )
        return encoders, middle_blks, downs, width

    def build_decoders(self, width, dec_blk_nums):
        ups = nn.ModuleList()
        decoders = nn.ModuleList()
        for num in dec_blk_nums:
            ups.append(
                nn.Sequential(
                    nn.Conv2d(width, width * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            width = width // 2
            decoders.append(
                nn.Sequential(
                    *[NAFBlock(width) for _ in range(num)]
                )
            )

        padder_size = 2 ** len(self.encoders)
        return ups, decoders, padder_size, width

    def forward(self, inp, pattern=None):
        B, C, H, W = inp.shape
        self.HighFrequency = torch.ones(B, C, int(H/2), int(W/2))
        inp = self.check_image_size(inp)

        x = self.intro(inp) # [b,width,h,w]
        if self.usePattern:
            featPattern = self.introPattern(pattern)
            x = torch.cat([x, featPattern.broadcast_to(B,featPattern.shape[1], H, W)], dim = 1)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x) # [8,512,16,16]
        nx = x
        x = x

        for decoder, up, enc_skip in zip(self.ndecoders, self.nups, encs[::-1]):
            nx = up(nx)
            nx = nx + enc_skip
            nx = decoder(nx)

        nx = self.nending(nx)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        
        if self.res:
            nx = nx + inp
            x = x + inp
        if self.tanh:
            nx = self.tanh(nx)
            x = self.tanh(x)
        else:
            x = x.clamp(-1+1e-6, 1-1e-6)
        output = torch.cat([nx, x], dim=1)
        return self.HighFrequency, output

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

@ARCH_REGISTRY.register()
class NAFNetHF(nn.Module):

    def __init__(self, in_channel=3, out_channel=10, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], res=False, tanh=True, **kwargs):
        super().__init__()

        self.res = res
        self.tanh = nn.Tanh() if tanh else tanh
        self.usePattern = kwargs.get('usePattern', False)
        if self.usePattern:
            self.intro = nn.Conv2d(in_channels=in_channel, out_channels=int(width*3/4), kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
            self.introPattern = self.build_intropattern(int(width/4))
        else:
            self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
            self.intro2 = nn.Conv2d(in_channels=in_channel-1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.encoders, self.middle_blks, self.downs, width = self.build_encoders(width, enc_blk_nums, middle_blk_num)
        width = width//2
        self.ups, self.decoders, self.padder_size, width = self.build_decoders(width, dec_blk_nums)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

    def build_intropattern(self, width):
        introPattern = []
        introPattern.append(nn.Conv2d(in_channels=3, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width))
        introPattern.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width))
        return nn.Sequential(*introPattern)

    def de_gamma(self,img):
        image = img**2.2
        image = torch.clip(image, min=0.0, max=1.0)
        return image

    def build_encoders(self, width, enc_blk_nums, middle_blk_num):
        encoders = nn.ModuleList()
        downs = nn.ModuleList()
        for num in enc_blk_nums:
            encoders.append(
                nn.Sequential(
                    *[NAFBlock(width) for _ in range(num)]
                )
            )
            downs.append(
                nn.Conv2d(width, 2*width, 2, 2)
            )
            width = width * 2
        width = width * 2 # two branch cat
        middle_blks = \
            nn.Sequential(
                *[NAFBlock(width) for _ in range(middle_blk_num)],
                nn.Conv2d(width, width//2, 1, bias=False)
            )
        return encoders, middle_blks, downs, width

    def up(self, width):
        ups = nn.ModuleList()
        ups.append(
                nn.Sequential(
                    nn.Conv2d(width, width // 2, 1, bias=False)
                )
        )
        return ups

    def build_decoders(self, width, dec_blk_nums):
        ups = nn.ModuleList()
        decoders = nn.ModuleList()
        for num in dec_blk_nums:
            ups.append(
                nn.Sequential(
                    nn.Conv2d(width, width * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            width = width // 2
            decoders.append(
                nn.Sequential(
                    *[NAFBlock(width) for _ in range(num)]
                )
            )

        padder_size = 2 ** len(self.encoders)
        return ups, decoders, padder_size, width

    def forward(self, inp, pattern=None):
        B, C, H, W = inp.shape # inp range
        self.HighFrequency = torch.ones(B, C, int(H/2), int(W/2))
        self.inputs_bands, self.dec = materialmodifier_L6.Show_subbands(self.de_gamma((inp + 1.0)/2.0), Logspace=True)
        self.inputs_bands = self.inputs_bands[:,3:5,:,:]
        input_bands = self.inputs_bands

        input_bands = self.check_image_size(input_bands)
        inp = self.check_image_size(inp)

        x2 = self.intro2(input_bands)
        x = self.intro(inp)

        if self.usePattern:
            featPattern = self.introPattern(pattern)
            x = torch.cat([x, featPattern.broadcast_to(B,featPattern.shape[1], H, W)], dim = 1)
        
        encs = []
        encs2 = []

        for encoder, down in zip(self.encoders, self.downs):
            x2 = encoder(x2)
            encs2.append(x2)
            x2 = down(x2)
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(torch.cat([x, x2], dim=1)) # [1024]->512

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x) # 256
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        if self.res:
            x = x + inp
        if self.tanh:
            x = self.tanh(x)
        else:
            x = x.clamp(-1+1e-6, 1-1e-6)
        return self.HighFrequency, x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

@ARCH_REGISTRY.register()
class MaterialNAFNet(NAFNet):
    def __init__(self, in_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], res=False, tanh=True, **kwargs):
        super(NAFNet, self).__init__()
        outputMapping = kwargs.get('outputMapping')
        self.seperateDecoders = kwargs.get('seperateDecoders', False)
        self.res = res
        self.tanh = nn.Tanh() if tanh else tanh
        self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.encoders, self.middle_blks, self.downs, width = self.build_encoders(width, enc_blk_nums, middle_blk_num)
        self.ending = nn.ModuleDict()
        if not self.seperateDecoders:
            self.ups = nn.ModuleDict()
            self.decoders = nn.ModuleDict()
            self.ups, self.decoders, padder_size, finalWidth = self.build_decoders(width, dec_blk_nums)
        else:
            for item in outputMapping.items():
                ups, decoders, padder_size, finalWidth = self.build_decoders(width, dec_blk_nums)
                self.ups.add_module(item[0], ups)
                self.decoders.add_module(item[0], decoders)
        for item in outputMapping.items():
            self.ending.add_module(item[0], nn.Conv2d(in_channels=finalWidth, out_channels=item[1]
                , kernel_size=3, padding=1, stride=1, groups=1, bias=True))
        self.padder_size = padder_size
        
    def forward(self, inp, pattern=None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        midX = self.middle_blks(x)
        result = []
        for i, key in enumerate(self.ending):
            if self.seperateDecoders:
                x = midX.clone()
                decoders = self.decoders[key]
                ups = self.ups[key]
                for decoder, up, enc_skip in zip(decoders, ups, encs[::-1]):
                    x = up(x)
                    x = x + enc_skip
                    x = decoder(x)
            elif i==0:
                for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
                    x = up(x)
                    x = x + enc_skip
                    x = decoder(x)

            resX = self.ending[key](x)
            if self.res:
                resX = resX + inp
            if self.tanh:
                resX = self.tanh(resX)
            else:
                if key == 'Roughness':
                    resX = resX.clamp(-1+1e-6, 1-1e-6)
                else:
                    resX = resX.clamp(-1, 1)

            result.append(resX)
        result = torch.cat(result, dim=1)
        return result[:, :, :H, :W]

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == "__main__":
    img_channel = 10
    width = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    print('enc blks', enc_blks, 'middle blk num', middle_blk_num, 'dec blks', dec_blks, 'width', width)

    # using('start . ')
    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    model.eval()
    print(model)
    input = torch.randn(4, 3, 256, 256)
    # input = torch.randn(1, 3, 32, 32)
    y = model(input)
    print(y.size())

    from thop import profile

    flops, params = profile(model=model, inputs=(input,))
    print('Model:{:.2f} GFLOPs and {:.2f}M parameters'.format(flops / 1e9, params / 1e6))
