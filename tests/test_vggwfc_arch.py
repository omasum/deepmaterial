from torchsummary import summary
from deepmaterial.archs.U_NETwoHF_arch import U_NETwoHF

myNet = U_NETwoHF()
myNet = myNet.to('cuda')
summary(myNet,(3,256,256),8)

