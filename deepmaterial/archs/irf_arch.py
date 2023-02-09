import torch
from torch import nn as nn

from deepmaterial.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class IRF(nn.Module):

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 encode_layer=[],
                 midcode_layer=[],
                 decode_layer=[],
                 attn_layer=[],
                 attn_fusion=False):
        super(IRF, self).__init__()
        self.inc = num_in_ch
        self.outc = num_out_ch
        self.ec = encode_layer
        self.mc = midcode_layer
        self.dc = decode_layer
        self.encode = []
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.first_fc = nn.Linear(num_in_ch, self.ec[0][0])
        self.attn_fusion=attn_fusion
        
        for item in self.ec:
            h,c = item
            self.encode.append(nn.Linear(h, c))
            self.encode.append(self.lrelu)
        self.encode = nn.Sequential(*self.encode)
        if attn_fusion:
            self.attn = []
            for item in attn_layer:
                h,c = item
                self.attn.append(nn.Linear(h, c))
                self.attn.append(self.lrelu)
            self.attn.pop(-1)
            self.soft = nn.Softmax(dim=-2)
            self.attn = nn.Sequential(*self.attn)
        
        self.midcode = []
        for item in self.mc:
            h,c = item
            self.midcode.append(nn.Linear(h, c))
            self.midcode.append(self.lrelu)
        self.mid_max = True if len(self.midcode) > 0 else False
        self.midcode = nn.Sequential(*self.midcode)
        
        self.max_pool = torch.max
        self.decoder = []
        for item in self.dc:
            h,c = item
            self.decoder.append(nn.Linear(h,c))
            self.decoder.append(self.lrelu)
        self.decoder = nn.Sequential(*self.decoder)
        self.final_fc = nn.Linear(c, num_out_ch)
        self.tanh = nn.Tanh()
        # self.drop = nn.Dropout(inplace=True)

    def forward(self, x, mask=None):
        # x: assembled measurements, shape is (b, n, c), b is class tag
        # mask: valid measurements, shape is (b, n)
        feat = self.lrelu(self.first_fc(x))
        feat = self.encode(feat)
        # if self.mid_max:
        #     feat_rand = feat[:,torch.randperm(feat.size(-2))]
        if mask is not None:
            feat = feat*(mask.unsqueeze(-1))
        if self.attn_fusion:
            attn_weights = self.soft(self.attn(feat))
            attn = attn_weights * feat
            if mask is not None:
                attn = attn*(mask.unsqueeze(-1))
            feat_global = torch.sum(attn, dim=-2)
        else:
            feat_global = self.max_pool(feat, dim=-2)[0]
        if self.mid_max:
            feat = self.midcode(torch.cat([feat, feat_global.unsqueeze(-2).broadcast_to(*feat.shape)], dim=-1))
            feat_global = self.max_pool(feat, dim=-2)[0]
        de_feat = self.decoder(feat_global)
        output = self.tanh(self.final_fc(de_feat))

        return output
