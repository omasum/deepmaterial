import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

class Deconv(nn.Module):

	def __init__(self,input_channel,output_channel):

		super(Deconv,self).__init__()
		## upsampling method (non-deterministic in pytorch)
		# self.upsampling=nn.Upsample(scale_factor=2, mode='nearest')

		self.temp_conv1 = nn.Conv2d(input_channel,output_channel,4,stride=1,bias=False)
		self.temp_conv2 = nn.Conv2d(output_channel,output_channel,4,stride=1,bias=False)

		# realize same padding in tensorflow
		self.padding=nn.ConstantPad2d((1, 2, 1, 2), 0)

	def forward(self,input):

		# print('Deco input shape,',input.shape[1])
		# Upsamp=self.upsampling(input)
		## hack upsampling method to make is deterministic
		Upsamp = input[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(input.size(0), input.size(1), input.size(2)*2, input.size(3)*2)

		out=self.temp_conv1(self.padding(Upsamp))
		out=self.temp_conv2(self.padding(out))

		# print('output shape,',out.shape)
		return out