
from deepmaterial.utils.registry import ARCH_REGISTRY
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from deepmaterial.archs.surfacenet_arch import SurfaceNet, upsample_conv, same_conv
from torchvision.models.resnet import resnet50 as resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch
    
@ARCH_REGISTRY.register()
class MSANNet(nn.Module):
    padding = 32
    out_feat = 256

    def __init__(self, **opt):
        super(MSANNet, self).__init__()
        self.opt = opt
        self.inputPattern = opt.get('inputPattern', False)
        channel = self.initEncoders()
        self.initDecoders(channel, 3*opt.get('nImgs', 1))
    
    def initEncoders(self):
        encoder = resnet(replace_stride_with_dilation=[
                          False, True, True], pretrained=True)

        self.encoder = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,

            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )
        inchannel = 2048 * self.opt.get('nImgs', 1)
        if self.inputPattern:
            self.lightEncoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 256, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 1024, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(1024, 2048, 3, 2, 1),
                nn.LeakyReLU(0.2)
            )
            inchannel += 2048
        return inchannel

    def initDecoders(self, inchannel, imgChannel):
        self.bottleneck = DeepLabHead(inchannel, 256)
        self.decoder = nn.ModuleList([
            upsample_conv(in_channels=256+imgChannel,
                          out_channels=self.out_feat),
            upsample_conv(in_channels=256+imgChannel,
                          out_channels=self.out_feat),
            upsample_conv(in_channels=256+imgChannel,
                          out_channels=self.out_feat),
            same_conv(in_channels=256+imgChannel,
                      out_channels=self.out_feat)
        ])

        self.heads = nn.ModuleDict()
        # create one head for each requested map
        self.outputMapping = self.opt["outputMapping"]
        for item in self.outputMapping.items():
            self.heads.add_module(item[0], self.__make_head(        
                self.out_feat, item[1]))

    def __make_head(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        
    def upsampleFeat(self, feat, x):
        feat = self.bottleneck(feat)
        # feat = self.conv(feat)
        for layer in self.decoder:
            pooled_x = F.interpolate(
                x, feat.shape[-2], mode="bilinear", align_corners=False)
            feat = layer(torch.cat((feat, pooled_x), 1))
        out = feat
        return out

    def feat2Map(self, out):
        maps = []
        for key in self.heads.keys():
            out_map = self.heads[key](out)
            out_map = out_map.clamp(-1, 1)
            maps.append(out_map)
        maps = torch.cat(maps,dim=1)
        return maps

    def forward(self, x, pattern = None):
        feat = self.encoder(x)
        if self.inputPattern:
            pattern = pattern.broadcast_to(feat.shape[0], *pattern.shape[1:])
            patternFeat = self.lightEncoder(pattern)
            feat = torch.cat([feat, patternFeat], dim=1)
        feat = self.upsampleFeat(feat, x)
        maps = self.feat2Map(feat)
        return maps

@ARCH_REGISTRY.register()
class MSANTwoStreamNet(MSANNet):
    def __init__(self, **opt):
        opt['nImgs'] = 2
        super().__init__(**opt)

    def upsampleFeat(self, feat, x):
        feat = self.bottleneck(feat)
        # feat = self.conv(feat)
        for layer in self.decoder:
            pooled_x1 = F.interpolate(
                x[0], feat.shape[-2], mode="bilinear", align_corners=False)
            pooled_x2 = F.interpolate(
                x[1], feat.shape[-2], mode="bilinear", align_corners=False)
            feat = layer(torch.cat((feat, pooled_x1, pooled_x2), 1))
        return feat

    def forward(self, x, pattern):
        feat = []
        patternFeat = []
        for xx, p in zip(x, pattern):
            feat.append(self.encoder(xx))
            if self.inputPattern:
                patternFeat.append(self.lightEncoder(p))
        feat = torch.cat(feat,1)
        if self.inputPattern:
            patternFeat = torch.max(torch.stack(patternFeat,0),dim=0,keepdim=False)[0]
            feat = torch.cat([feat,patternFeat], dim=1)
        out = self.upsampleFeat(feat, x)
        maps = self.feat2Map(out)
        return maps