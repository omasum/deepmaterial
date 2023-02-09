import torch
import torch.nn as nn

from deepmaterial.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class distNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv2 = nn.Conv2d(112,16,3,1,1)
        self.conv3 = nn.Conv2d(16,16,3,1,1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.fc = nn.Sequential(
            nn.Linear(16, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256,16)
        )
        # upsample
        self.upconv1 = nn.Conv2d(16, 16 * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)
        
    
    def forward(self, x):
        b,t,c,h,w = x.shape

        feat = self.lrelu(self.conv1(x.view(-1,c,h,w)))
        feat = feat.view(b,t,-1, h, w).view(b,-1,h,w)

        feat = self.lrelu(self.conv2(feat))
        out = self.conv3(feat)

        out = self.fc(out.permute(0,2,3,1).contiguous())
        out = out.permute(0,3,1,2).contiguous()
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        
        return out