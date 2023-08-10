import os
import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import math
from deepmaterial.archs.autoencoder import Deconv
from deepmaterial.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class NewVANet(nn.Module):
	def __init__(self,input_channel,output_channel, rough_channel):
		super(NewVANet,self).__init__()
		
		self.rough_nc=rough_channel

		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_channel,64,4,2,1,bias=False)
		self.conv2 = nn.Conv2d(64,128,4,2,1,bias=False)
		self.conv3 = nn.Conv2d(128,256,4,2,1,bias=False)
		self.conv4 = nn.Conv2d(256,512,4,2,1,bias=False)
		self.conv5 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv6 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv7 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv8 = nn.Conv2d(512,512,4,2,1,bias=False)

		#decoder(diff)
		self.deconv1_diff = Deconv(512, 512)
		self.deconv2_diff = Deconv(1024, 512)
		self.deconv3_diff = Deconv(1024, 512)
		self.deconv4_diff = Deconv(1024, 512)
		self.deconv5_diff = Deconv(1024, 256)
		self.deconv6_diff = Deconv(512, 128)
		self.deconv7_diff = Deconv(256, 64)
		self.deconv8_diff = Deconv(128, output_channel)

		#decoder(normal)
		self.deconv1_normal = Deconv(512, 512)
		self.deconv2_normal = Deconv(1024, 512)
		self.deconv3_normal = Deconv(1024, 512)
		self.deconv4_normal = Deconv(1024, 512)
		self.deconv5_normal = Deconv(1024, 256)
		self.deconv6_normal = Deconv(512, 128)
		self.deconv7_normal = Deconv(256, 64)
		self.deconv8_normal = Deconv(128, output_channel)

		#decoder(rough)
		self.deconv1_rough = Deconv(512, 512)
		self.deconv2_rough = Deconv(1024, 512)
		self.deconv3_rough = Deconv(1024, 512)
		self.deconv4_rough = Deconv(1024, 512)
		self.deconv5_rough = Deconv(1024, 256)
		self.deconv6_rough = Deconv(512, 128)
		self.deconv7_rough = Deconv(256, 64)
		self.deconv8_rough = Deconv(128, rough_channel)

		#decoder(spec)
		self.deconv1_spec = Deconv(512, 512)
		self.deconv2_spec = Deconv(1024, 512)
		self.deconv3_spec = Deconv(1024, 512)
		self.deconv4_spec = Deconv(1024, 512)
		self.deconv5_spec = Deconv(1024, 256)
		self.deconv6_spec = Deconv(512, 128)
		self.deconv7_spec = Deconv(256, 64)
		self.deconv8_spec = Deconv(128, output_channel)

		self.sig = nn.Sigmoid()
		self.tan = nn.Tanh()


		self.leaky_relu = nn.LeakyReLU(0.2)

		# self.instance_normal1 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal2 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal3 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal4 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal5 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal6 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal7 = nn.InstanceNorm2d(512,affine=True)

		self.instance_normal_de_1_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_diff = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_diff = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_diff = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_normal = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_normal = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_normal = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_rough = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_rough = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_rough = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_spec = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_spec = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_spec = nn.InstanceNorm2d(64,affine=True)

		self.dropout = nn.Dropout(0.5)


	def forward(self, input):

		# [batch,64,h/2,w/2]
		encoder1 = self.conv1(input) #local network
		# [batch,128,h/4,w/4]        
		encoder2 = self.instance_normal2(self.conv2(self.leaky_relu(encoder1))) #local network
		# [batch,256,h/8,w/8]        
		encoder3 = self.instance_normal3(self.conv3(self.leaky_relu(encoder2))) #local network
		# [batch,512,h/16,w/16]        
		encoder4 = self.instance_normal4(self.conv4(self.leaky_relu(encoder3))) #local network
		# [batch,512,h/32,w/32]        
		encoder5 = self.instance_normal5(self.conv5(self.leaky_relu(encoder4))) #local network
		# [batch,512,h/64,w/64]        
		encoder6 = self.instance_normal6(self.conv6(self.leaky_relu(encoder5))) #local network
		# [batch,512,h/128,w/128]        
		encoder7 = self.instance_normal7(self.conv7(self.leaky_relu(encoder6))) #local network
		# [batch,512,h/256,w/256]
		encoder8 = self.conv8(self.leaky_relu(encoder7)) # local


		################################## decoder (diff) #############################################
		# [batch,512,h/128,w/128]
		decoder1_diff = self.instance_normal_de_1_diff(self.deconv1_diff(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_diff = torch.cat((self.dropout(decoder1_diff), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_diff = self.instance_normal_de_2_diff(self.deconv2_diff(self.leaky_relu(decoder1_diff)))
		# [batch,1024,h/64,w/64]
		decoder2_diff = torch.cat((self.dropout(decoder2_diff), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_diff = self.instance_normal_de_3_diff(self.deconv3_diff(self.leaky_relu(decoder2_diff)))
		# [batch,1024,h/32,w/32]
		decoder3_diff = torch.cat((self.dropout(decoder3_diff), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_diff = self.instance_normal_de_4_diff(self.deconv4_diff(self.leaky_relu(decoder3_diff)))
		# [batch,1024,h/16,w/16]
		decoder4_diff = torch.cat((decoder4_diff, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_diff = self.instance_normal_de_5_diff(self.deconv5_diff(self.leaky_relu(decoder4_diff)))
		# [batch,512,h/8,w/8]
		decoder5_diff = torch.cat((decoder5_diff, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_diff = self.instance_normal_de_6_diff(self.deconv6_diff(self.leaky_relu(decoder5_diff)))
		# [batch,256,h/4,w/4]
		decoder6_diff = torch.cat((decoder6_diff, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_diff = self.instance_normal_de_7_diff(self.deconv7_diff(self.leaky_relu(decoder6_diff)))
		# [batch,128,h/2,w/2]
		decoder7_diff = torch.cat((decoder7_diff, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_diff = self.deconv8_diff(self.leaky_relu(decoder7_diff))

		diff = self.tan(decoder8_diff)
		# print(output.shape)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_normal = self.instance_normal_de_1_normal(self.deconv1_normal(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_normal = torch.cat((self.dropout(decoder1_normal), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_normal = self.instance_normal_de_2_normal(self.deconv2_normal(self.leaky_relu(decoder1_normal)))
		# [batch,1024,h/64,w/64]
		decoder2_normal = torch.cat((self.dropout(decoder2_normal), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_normal = self.instance_normal_de_3_normal(self.deconv3_normal(self.leaky_relu(decoder2_normal)))
		# [batch,1024,h/32,w/32]
		decoder3_normal = torch.cat((self.dropout(decoder3_normal), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_normal = self.instance_normal_de_4_normal(self.deconv4_normal(self.leaky_relu(decoder3_normal)))
		# [batch,1024,h/16,w/16]
		decoder4_normal = torch.cat((decoder4_normal, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_normal = self.instance_normal_de_5_normal(self.deconv5_normal(self.leaky_relu(decoder4_normal)))
		# [batch,512,h/8,w/8]
		decoder5_normal = torch.cat((decoder5_normal, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_normal = self.instance_normal_de_6_normal(self.deconv6_normal(self.leaky_relu(decoder5_normal)))
		# [batch,256,h/4,w/4]
		decoder6_normal = torch.cat((decoder6_normal, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_normal = self.instance_normal_de_7_normal(self.deconv7_normal(self.leaky_relu(decoder6_normal)))
		# [batch,128,h/2,w/2]
		decoder7_normal = torch.cat((decoder7_normal, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_normal = self.deconv8_normal(self.leaky_relu(decoder7_normal))

		normal = self.tan(decoder8_normal)
		# print(output.shape)
	 
		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_rough = self.instance_normal_de_1_rough(self.deconv1_rough(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_rough = torch.cat((self.dropout(decoder1_rough), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_rough = self.instance_normal_de_2_rough(self.deconv2_rough(self.leaky_relu(decoder1_rough)))
		# [batch,1024,h/64,w/64]
		decoder2_rough = torch.cat((self.dropout(decoder2_rough), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_rough = self.instance_normal_de_3_rough(self.deconv3_rough(self.leaky_relu(decoder2_rough)))
		# [batch,1024,h/32,w/32]
		decoder3_rough = torch.cat((self.dropout(decoder3_rough), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_rough = self.instance_normal_de_4_rough(self.deconv4_rough(self.leaky_relu(decoder3_rough)))
		# [batch,1024,h/16,w/16]
		decoder4_rough = torch.cat((decoder4_rough, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_rough = self.instance_normal_de_5_rough(self.deconv5_rough(self.leaky_relu(decoder4_rough)))
		# [batch,512,h/8,w/8]
		decoder5_rough = torch.cat((decoder5_rough, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_rough = self.instance_normal_de_6_rough(self.deconv6_rough(self.leaky_relu(decoder5_rough)))
		# [batch,256,h/4,w/4]
		decoder6_rough = torch.cat((decoder6_rough, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_rough = self.instance_normal_de_7_rough(self.deconv7_rough(self.leaky_relu(decoder6_rough)))
		# [batch,128,h/2,w/2]
		decoder7_rough = torch.cat((decoder7_rough, encoder1), 1)

		# [batch,_out_c,h,w]
		decoder8_rough = self.deconv8_rough(self.leaky_relu(decoder7_rough))

		rough = self.tan(decoder8_rough)
		# print(output.shape)
		if self.rough_nc==1:
			rough=rough.repeat(1,3,1,1)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_spec = self.instance_normal_de_1_spec(self.deconv1_spec(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_spec = torch.cat((self.dropout(decoder1_spec), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_spec = self.instance_normal_de_2_spec(self.deconv2_spec(self.leaky_relu(decoder1_spec)))
		# [batch,1024,h/64,w/64]
		decoder2_spec = torch.cat((self.dropout(decoder2_spec), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_spec = self.instance_normal_de_3_spec(self.deconv3_spec(self.leaky_relu(decoder2_spec)))
		# [batch,1024,h/32,w/32]
		decoder3_spec = torch.cat((self.dropout(decoder3_spec), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_spec = self.instance_normal_de_4_spec(self.deconv4_spec(self.leaky_relu(decoder3_spec)))
		# [batch,1024,h/16,w/16]
		decoder4_spec = torch.cat((decoder4_spec, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_spec = self.instance_normal_de_5_spec(self.deconv5_spec(self.leaky_relu(decoder4_spec)))
		# [batch,512,h/8,w/8]
		decoder5_spec = torch.cat((decoder5_spec, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_spec = self.instance_normal_de_6_spec(self.deconv6_spec(self.leaky_relu(decoder5_spec)))
		# [batch,256,h/4,w/4]
		decoder6_spec = torch.cat((decoder6_spec, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_spec = self.instance_normal_de_7_spec(self.deconv7_spec(self.leaky_relu(decoder6_spec)))
		# [batch,128,h/2,w/2]
		decoder7_spec = torch.cat((decoder7_spec, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_spec = self.deconv8_spec(self.leaky_relu(decoder7_spec))

		spec = self.tan(decoder8_spec)

		output=torch.cat((normal,diff,rough,spec),1)

		# print('shape: ',output.shape)

		return None, output