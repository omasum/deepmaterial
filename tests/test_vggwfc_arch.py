from torchsummary import summary
from deepmaterial.archs.wfc_sf_arch import wfc_sf_arch

myNet = wfc_sf_arch('VGG11')
myNet = myNet.to('cuda')
summary(myNet,(3,256,256),8)

