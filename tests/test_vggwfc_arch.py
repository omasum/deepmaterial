# from torchsummary import summary
# from deepmaterial.archs.U_NETwoHF_arch import U_NETwoHF

# myNet = U_NETwoHF()
# myNet = myNet.to('cuda')
# summary(myNet,(3,256,256),8)

from deepmaterial.archs.NAFNet_arch import NAFNet
import torch
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

