from torch import nn as nn

from deepmaterial.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator128(nn.Module):
    """VGG style discriminator with input size 128 x 128.

    It is used to train SRGAN and ESRGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch, num_feat):
        super(VGGStyleDiscriminator128, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 128 and x.size(3) == 128, (f'Input spatial size must be 128x128, '
                                                       f'but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

    
@ARCH_REGISTRY.register()
class PatchDiscriminator(nn.Module):
    def __init__(self,**opt):
        super(PatchDiscriminator, self).__init__()
        # in_channels is computed as the sum of channels per map + the channesl for the rendering (3)
        in_channels = 13

        self.main = NLayerDiscriminator(in_channels=in_channels, n_layers=4)

    def forward(self, x):
        out = self.main(x)

        return out

@ARCH_REGISTRY.register()
class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        # in_channels is computed as the sum of channels per map + the channesl for the rendering (3)
        # in_channels = 3 + sum([textures_mapping[x] for x in texture_maps])
        in_channels = 13

        n_layers = 6

        self.main = NLayerDiscriminator(in_channels=in_channels, n_layers=n_layers, final_classifier=False)

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(2),
            nn.Conv2d(512, 1, kernel_size=2)
        )


    def forward(self, x):
        out = self.main(x)
        out = self.classifier(out)
        return out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_features=64, n_layers=3, norm_layer=nn.BatchNorm2d, final_classifier=True, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1

        sequence = [
            nn.Conv2d(in_channels, base_features, kernel_size=kernel_size,
                      stride=2, padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(base_features * nf_mult_prev, base_features * nf_mult,
                          kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(base_features * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(base_features * nf_mult_prev, base_features * nf_mult,
                      kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_layer(base_features * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        if final_classifier:
            sequence += [nn.Conv2d(base_features * nf_mult, 1,
                               kernel_size=kernel_size, stride=1, padding=padding)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
