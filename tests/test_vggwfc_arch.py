from torchsummary import summary
from deepmaterial.archs.vgg_wfc_arch import VGG_wfc

myNet = VGG_wfc('VGG11')
myNet = myNet.to('cuda')
summary(myNet,(3,256,256),8)