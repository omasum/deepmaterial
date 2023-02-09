from time import time
from tracemalloc import start
from deepmaterial.utils.render_util import PlanarSVBRDF, RectLighting, PolyRender, viewDistance
import cv2 as cv
import torch, torchvision
import numpy as np
from deepmaterial.losses import build_loss

brdfArgs = {}
brdfArgs['nbRendering'] = 1
brdfArgs['size'] = 256
brdfArgs['order'] = 'pndrs'
brdfArgs['toLDR'] = False
brdfArgs['lampIntensity'] = 3
path = '/home/sda/svBRDFs/testBlended/0000005;PolishedMarbleFloor_01Xmetal_bumpy_squares;0Xdefault.png'
device = 'cuda'
printFreq = 100

svbrdf = PlanarSVBRDF(brdfArgs)
img = cv.imread(path)[:, :, ::-1] / 255
svbrdf.get_svbrdfs(img)
svbrdf.to(device)
brdfArgs['textureMode'] = 'Torch'
brdfArgs['fetchMode'] = 'NvDiff'
brdfArgs['nLod'] = 8
brdfArgs['texture'] = 'tmp/0.png'
rect = RectLighting(brdfArgs, [0, 0.0, viewDistance], [1, 0, 0], [0, 1, 0], 2.0, 2.0, device=device)
rect.verticesToDevice(device)
renderer = PolyRender(brdfArgs, lighting=rect, device=device)
surface = renderer.generate_surface(brdfArgs['size']).to(device)
nbInputs = 100
imgs = []
viewPoss = []
for i in range(nbInputs):
    viewDistance = np.random.uniform(1, 5)
    viewPos = renderer.lighting.fullRandomLightsSurface(batch=1, lightDistance=viewDistance).to(device)
    viewPoss.append(viewPos)
    img = renderer.render(svbrdf, view_pos = viewPos, obj_pos=surface)
    imgs.append(img)
gtImgs = torch.stack(imgs, dim=0)
torchvision.utils.save_image(
    gtImgs**0.4545, f'tmp/gt.png', nrow=10, normalize=False)

initPattern = torch.empty_like(rect.tex, device=device).uniform_(-1,1)
initPattern.requires_grad = True
optimizer = torch.optim.Adam([initPattern], lr=1e-2)
l2Loss = build_loss({'type':'MSELoss','reduction':'mean'}).to(device)
totalIter = 10000
textureMode = 'Torch'
print('Start optimizing....')
start_time = time()
for i in range(totalIter):
    predImgs = []
    optimizer.zero_grad()
    rect.initTexture(torch.sigmoid(initPattern), textureMode)
    for j in range(nbInputs):
        viewPos = viewPoss[j]
        img = renderer.render(svbrdf, view_pos=viewPos, obj_pos=surface)
        predImgs.append(img)
    predImgs = torch.stack(predImgs, dim=0)
    
    loss = l2Loss(gtImgs, predImgs)
    # for i in rect.lod:
    #     i.retain_grad()
    loss.backward()
    optimizer.step()
    if (i+1) % printFreq == 0:
        print('step: ', i, ' loss: ', loss, 'cost time: ', time() - start_time)
        start_time = time()
        torchvision.utils.save_image(
            predImgs**0.4545, f'tmp/pred{i}.png', nrow=10, normalize=False)
        torchvision.utils.save_image(
            torch.sigmoid(initPattern), f'tmp/pattern{i}.png', nrow=10, normalize=False)
        
