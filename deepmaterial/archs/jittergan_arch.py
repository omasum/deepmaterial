import math
import random
import torch
from torch import nn
from torch.nn import functional as F

from deepmaterial.utils.registry import ARCH_REGISTRY
from deepmaterial.archs.stylegan2_arch import *

import torch

@ARCH_REGISTRY.register()
class JitterGANGenerator(nn.Module):
    """JitterGAN Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 constant_init=True,
                 narrow=1):
        super(JitterGANGenerator, self).__init__()
        # Style MLP layers
        self.num_style_feat = num_style_feat
        self.constant_init = constant_init
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.append(
                EqualLinear(
                    num_style_feat, num_style_feat, bias=True, bias_init_val=0, lr_mul=lr_mlp,
                    activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        self.channels = channels

        if self.constant_init:
            self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(
            channels['4'],
            channels['4'],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
            resample_kernel=resample_kernel)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, brdf=True, upsample=False, resample_kernel=resample_kernel)
        self.tanh = nn.Tanh()
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels['4']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode='upsample',
                    resample_kernel=resample_kernel,
                ))
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                    resample_kernel=resample_kernel))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, brdf=True, upsample=True, resample_kernel=resample_kernel))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        noises = [torch.randn(1, 1, 4, 4, device=self.device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=self.device))

        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = torch.randn(num_latent, self.num_style_feat, device=self.device)
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(self,
                styles,
                init=None,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2Generator.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style.
                Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is
                False. Default: True.
            truncation (float): TODO. Default: 1.
            truncation_latent (Tensor | None): TODO. Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
                Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latent with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        if self.constant_init:
            out = self.constant_input(latent.shape[0])
        else:
            out = init
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip
        image = self.tanh(image)
        if return_latents:
            return image, latent
        else:
            return image, None


@ARCH_REGISTRY.register()
## 生成器 U-Net（输入照片为256*256） ##
class Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=[1,2,4,8,8,8], kSize=3, ngf=64):
        """
        定义生成器的网络结构
        :param in_ch: 输入数据的通道数
        :param out_ch: 输出数据的通道数
        :param ngf: 第一层卷积的通道数 number of generator's first conv filters
        """
        super(Encoder, self).__init__()
        # 下面的激活函数都放在下一个模块的第一步 是为了skip-connect方便
        self.outC = []
        self.encoder = nn.ModuleList()
        lastoutC = 1
        for i, outC in enumerate(out_ch):
            self.outC.append(outC*ngf)
            if i != 0:
                self.encoder.append(nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(lastoutC*ngf, outC*ngf, kernel_size=kSize, stride=2, padding=kSize//2),
                    nn.InstanceNorm2d(outC)
                ))
            else:
                self.encoder.append(nn.Sequential(
                    nn.Conv2d(in_ch, outC*ngf, kernel_size=kSize, stride=2, padding=kSize//2),
                    # 输入图片已正则化 不需BatchNorm
                    nn.InstanceNorm2d(outC*ngf)
                ))
            lastoutC = outC

    def forward(self, X):
        """
        生成器模块前向传播
        :param X: 输入生成器的数据
        :return: 生成器的输出
        """
        # Encoder
        out_feat = []
        out = X
        for module in self.encoder:
            out = module(out)
            out_feat.append(out)
        return out_feat
