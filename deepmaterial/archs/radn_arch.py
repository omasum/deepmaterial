import numpy as np
import torch
import torch.nn as nn

from deepmaterial.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F

@ARCH_REGISTRY.register()
class RADN(nn.Module):
    def __init__(self, en_channels, de_channels, drop_layer) -> None: # 几层dropout
        super().__init__()

        #  encoder layers
        self.en_convs = []
        self.en_fcs = []
        self.en_g2e_fcs = []
        self.en_ins_norms=[]
        in_channels = en_channels[0:-1]
        out_channels = en_channels[1:]
        fc_out_channels = en_channels[2:]
        fc_out_channels.append(out_channels[-1])
        for i, (inC, outC, fc_outC) in enumerate(zip(in_channels, out_channels, fc_out_channels)):
            if i == 0:
                self.en_convs.append(nn.Conv2d(inC, outC, 4, 2, padding=[1,1], bias=False))
                self.en_fcs.append(nn.Sequential(
                    nn.Linear(inC, fc_outC),
                    nn.SELU(inplace=True)
                ))
            elif i == len(in_channels)-1:
                self.en_convs.append(nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(inC, outC, 4, 2, padding=[1,1], bias=False)
                ))
                self.en_fcs.append(nn.Sequential(
                    nn.Linear(outC+last_outC, fc_outC),
                    nn.SELU(inplace=True)
                ))
                self.en_g2e_fcs.append(nn.Linear(outC, outC, False))
            else:
                self.en_convs.append(nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(inC, outC, 4, 2, padding=[1,1], bias=False)
                ))
                self.en_ins_norms.append(nn.InstanceNorm2d(outC, affine=True))
                self.en_fcs.append(nn.Sequential(
                    nn.Linear(outC+last_outC, fc_outC),
                    nn.SELU(inplace=True)
                ))
                self.en_g2e_fcs.append(nn.Linear(outC, outC, False))
            
            last_outC = fc_outC
        self.en_convs = nn.Sequential(*self.en_convs)
        self.en_fcs = nn.Sequential(*self.en_fcs)
        self.en_g2e_fcs = nn.Sequential(*self.en_g2e_fcs)
        self.en_ins_norms = nn.Sequential(*self.en_ins_norms)

        # decoder layers
        in_channels = de_channels[0:-1]
        out_channels = de_channels[1:]
        fc_out_channels = de_channels[1:] # 之前的输出通道不对
        self.de_convs = []
        self.de_fcs = []
        self.de_g2e_fcs = []
        self.de_ins_norms=[]
        self.drop_layer = drop_layer
        self.drop = nn.Dropout(inplace=True)
        for i, (inC, outC, fc_outC) in enumerate(zip(in_channels, out_channels, fc_out_channels)):
            if i != 0 and i < len(self.en_convs):
                inC = inC *2
            if i != len(in_channels)-1:
                self.de_ins_norms.append(nn.InstanceNorm2d(outC, affine=True))
                self.de_fcs.append(nn.Sequential(
                    nn.Linear(outC+last_outC, fc_outC),
                    nn.SELU(inplace=True)
                ))
            self.de_convs.append(nn.Sequential(
                nn.Conv2d(inC, outC, 4, 1, padding='same', bias=False),
                nn.Conv2d(outC, outC, 4, 1, padding='same', bias=False)
            ))
            self.de_g2e_fcs.append(nn.Linear(last_outC, outC, False)) # 之前的输出通道不对
            last_outC = fc_outC

        self.de_convs = nn.Sequential(*self.de_convs)
        self.de_fcs = nn.Sequential(*self.de_fcs)
        self.de_g2e_fcs = nn.Sequential(*self.de_g2e_fcs)
        self.de_ins_norms = nn.Sequential(*self.de_ins_norms)

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.tanh = nn.Tanh()

        self.init_modules()

    def init_modules(self):
        module_list = [self.en_convs, self.en_fcs, self.en_g2e_fcs, self.en_ins_norms, self.de_convs, self.de_fcs, self.de_g2e_fcs, self.de_ins_norms]
        for module in module_list:
            if module == self.en_g2e_fcs or module == self.de_g2e_fcs:# g2e_fc的全连接层方差要乘一个0.01
                multiply = 0.01
            else:
                multiply = 1.0 
            for m in module.modules():
                if m == module:
                    continue
                if isinstance(m, nn.Sequential):
                    for msub in m.modules():
                        if msub == m:
                            continue
                        self.init_weights(msub)
                else:
                    self.init_weights(m, multiply) # 改动
    def init_weights(self, m, multiply = 1.0): # 改动
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
        if isinstance(m, nn.InstanceNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, np.sqrt(1.0/m.weight.data.shape[1])*multiply) # g2e_fc的全连接层方差要乘一个0.01
            if m.bias is not None:
                nn.init.normal_(m.bias, 0,0.002)

    def forward(self, x):
        layers = []
        mean = x.mean(dim=(2,3))
        for i, (conv, fc) in enumerate(zip(self.en_convs, self.en_fcs)):
            output = conv(x)
            if i != 0:
                mean = output.mean(dim=(2,3))
                mean = torch.cat([globalOutput, mean], dim=-1)
                if i != len(self.en_convs)-1:
                    output = self.en_ins_norms[i-1](output)
                b, c = globalOutput.shape
                output = output + self.en_g2e_fcs[i-1](globalOutput).view(b,c, 1, 1)
            globalOutput = fc(mean)
            x = output
            layers.append(x)
        n_encoder_layer=len(layers)
        
        for i, (deconv, g2e_fc) in enumerate(zip(self.de_convs, self.de_g2e_fcs)):
            if i != 0 and i < n_encoder_layer:
                x = torch.cat([x,layers[-(i+1)]], dim=1)
            x = self.lrelu(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            output = deconv(x)
            if i != len(self.de_convs)-1:
                mean = output.mean(dim=(2,3))
                mean = torch.cat([globalOutput, mean], dim=-1) # 之前写反了，本来和encoder里面是一致的，然后昨天查他源代码发现是globaloutput在前面，encoder改了，decoder忘记改了，估计影响很大
                output = self.de_ins_norms[i](output)
            b, c = output.shape[:2] # g2e_fc的输出是和output的通道数一致，不是和上一次的globalOutput一致
            output = output + g2e_fc(globalOutput).view(b,c, 1, 1)
            if i < self.drop_layer: # decoder前三层要dropout，概率为0.5
                output = self.drop(output)
            if i != len(self.de_convs)-1:
                globalOutput = self.de_fcs[i](mean)
            x = output
        x = self.tanh(x)
        
        return x