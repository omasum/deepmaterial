""" assemble parts to form the encoder and decoder """

from .unet_parts import *

class Encoder(nn.Module):
    def __init__(self, n_channels, layers=False, bilinear=False):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.layers = layers
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if self.layers:
            x5 = self.down4(x4)
            return x1, x2, x3, x4, x5
        else:
            return x1, x2, x3, x4

class subEncoder(nn.Module):
    def __init__(self, n_channels, layers=False, bilinear=False):
        super(subEncoder, self).__init__()
        self.n_channels = n_channels
        self.layers = layers
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if self.layers:
            x4 = self.down3(x3)
            return x1, x2, x3, x4
        else:
            return x1, x2, x3

class subDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(subDecoder, self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x4, x3, x2, x1):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class Decoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Decoder, self).__init__()
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x5, x4, x3, x2, x1):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        """ SE注意力机制,输入x。输入输出特征图不变
            1.squeeze: 全局池化 (batch,channel,height,width) -> (batch,channel,1,1) ==> (batch,channel)
            2.excitaton: 全连接or卷积核为1的卷积(batch,channel)->(batch,channel//reduction)-> (batch,channel) ==> (batch,channel,1,1) 输出y
            3.scale: 完成对通道维度上原始特征的标定 y = x*y 输出维度和输入维度相同

        :param channel: 输入特征图的通道数
        :param reduction: 特征图通道的降低倍数
        """
        super(SELayer, self).__init__()
        # 自适应全局平均池化,即，每个通道进行平均池化，使输出特征图长宽为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接的excitation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        # 卷积网络的excitation
        # 特征图变化：
        # (2,512,1,1) -> (2,512,1,1) -> (2,512,1,1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (batch,channel,height,width) (2,512,8,8)
        b, c, _, _ = x.size()
        # 全局平均池化 (2,512,8,8) -> (2,512,1,1) -> (2,512)
        y = self.avg_pool(x).view(b, c)
        # (2,512) -> (2,512//reducation) -> (2,512) -> (2,512,1,1)
        y = self.fc(y).view(b, c, 1, 1)
        # (2,512,8,8)* (2,512,1,1) -> (2,512,8,8)
        pro = x * y
        return x * y
