from torchsummary import summary
from deepmaterial.archs.vgg_dwt_arch import VGG_dwt_sf

myNet = VGG_dwt_sf('VGG11')
myNet = myNet.to('cuda')
summary(myNet,(3,256,256),8)