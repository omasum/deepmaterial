import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

tex = torch.load('/home/sda/klkjjhjkhjhg/DeepMaterial/experiments/Exp_0005(2)_MSAN_OptimizePattern_clip/models/pattern_150000.pth', 'cuda')/2+0.5
uv = torch.empty((1,256,256,2), device='cuda').uniform_(0,1)
uv.requires_grad = True
lod = torch.empty((1,256,256), device='cuda').uniform_(0,8)
mips = []
mip = tex.clone()
for i in range(10):
    mip = F.interpolate(mip, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
    mips.append(mip.permute(0,2,3,1).contiguous())

# tex = tex.permute(0,2,3,1).contiguous()
newtex = tex.permute(0,2,3,1).contiguous().detach().clone()

color = dr.texture(newtex, uv, mip=mips)

print('mipmap')