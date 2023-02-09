
import torch.nn as nn
import torch.nn.functional as F
import torch
from deepmaterial.utils.registry import ARCH_REGISTRY


def xavier_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

def conv_layer(in_channel,out_channel,k,stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    return nn.Conv2d(in_channel,out_channel,kernel_size = k,stride=stride,padding=pad)


def block(in_channel,out_channel,k,stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    
    return nn.Sequential(
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channel,out_channel,kernel_size = k,stride=stride,padding=pad),
        nn.Dropout2d(p=0.2,inplace=True)
        
    )

def dense_layer(in_channel,out_channel):
    return nn.Sequential(
            nn.Linear(in_channel,out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )

@ARCH_REGISTRY.register()
class Dense3D(nn.Module):
    def __init__(self, in_channel, out_channel, oh,ow,init_method='xavier',channel_num=16):
        super().__init__()
        h,w = oh//2,ow//2
        self.conv1 = conv_layer(in_channel,channel_num,3)
        self.conv2 = block(channel_num,channel_num,3)
        self.conv3 = block(2*channel_num,16,3)
        self.conv4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            conv_layer(3*channel_num,3*channel_num,1),
            nn.Dropout2d(p=0.2),
            nn.AvgPool2d(kernel_size=2,stride=2)
        )
        self.conv5 = block(3*channel_num,channel_num,3)
        self.conv6 = block(4*channel_num,channel_num,3)
        self.conv7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            conv_layer(5*channel_num,80,1)
        )
        
        self.dense1b = dense_layer(5*channel_num*h*w,128)
        self.dense2b = nn.Linear(128,out_channel)
        self.tanh = nn.Tanh()
        # self.sig= nn.Sigmoid()
        
        if init_method == 'xavier':
            self.apply(xavier_init)
            
    def forward(self,x0, mask):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        xc1 = torch.cat([x2,x1],dim=1)
        
        x3 = self.conv3(xc1)
        xc2 = torch.cat([x3,x2,x1],dim=1)
        
        x1 = self.conv4(xc2)
        x2 = self.conv5(x1)
        xc1 = torch.cat([x2,x1],dim=1)
        
        x3 = self.conv6(xc1)
        xc2 = torch.cat([x3,x2,x1],dim=1)
        
        x = self.conv7(xc2)
        x = x.view(x.shape[0],-1)
        
        x = self.dense1b(x)
        x = self.dense2b(x)
        
        # x = self.tanh(x)
        # x should be batch*3
        # x = F.normalize(x,2,1)
        
        return x 