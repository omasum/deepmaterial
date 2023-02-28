from torchsummary import summary
from deepmaterial.archs.vgg_wfc_arch_sf import VGG_wfc_sf

myNet = VGG_wfc_sf('VGG11')
myNet = myNet.to('cuda')
summary(myNet,(3,256,256),8)