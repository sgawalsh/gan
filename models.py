import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class disc_conv_net(nn.Module):
	def __init__(self, output: torch.Size):
		super(disc_conv_net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 1)
		self.out_activation = nn.Sigmoid()

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		return self.out_activation(self.fc2(x))
		
class gen_conv_net(nn.Module):
	def __init__(self, input: int, output: torch.Size):
		super(gen_conv_net, self).__init__()
		self.dense = nn.Linear(input, 784)
		self.activation = nn.Sigmoid()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding = 2)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding = 2)
		self.conv3 = nn.Conv2d(20, 1, kernel_size=5, padding = 2)


	def forward(self, x):
		x = self.activation(self.dense(x))
		x = x.reshape(-1, 1, 28, 28)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		return F.relu(self.conv3(x))
		
class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (nc) x 64 x 64
		)

	def forward(self, input):
		return self.main(input)
		
class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, input):
		return self.main(input)
		
class full_conv_gen_64(nn.Module):
	def __init__(self, input_channels: int = 100, out_channels: int = 3, ngf: int = 64):
		super(full_conv_gen_64, self).__init__()
		self.conv1 = nn.ConvTranspose2d(input_channels, ngf * 8, 4, 1, 0, bias=False)
		self.batch1 = nn.BatchNorm2d(ngf * 8)# state size. (ngf*8) x 4 x 4
		
		self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
		self.batch2 = nn.BatchNorm2d(ngf * 4)# state size. (ngf*4) x 8 x 8
		
		self.conv3 = nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False)
		self.batch3 = nn.BatchNorm2d(ngf * 2)# state size. (ngf*2) x 16 x 16
		
		self.conv4 = nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False)
		self.batch4 = nn.BatchNorm2d(ngf)# state size. (ngf) x 32 x 32
		
		self.conv5 = nn.ConvTranspose2d( ngf, out_channels, 4, 2, 1, bias=False)
		self.tanh = nn.Tanh()
		
	def forward(self, x):
		x = F.relu(self.batch1(self.conv1(x)))
		x = F.relu(self.batch2(self.conv2(x)))
		x = F.relu(self.batch3(self.conv3(x)))
		x = F.relu(self.batch4(self.conv4(x)))
		return self.tanh(self.conv5(x))
		
class full_conv_disc_64(nn.Module):
	def __init__(self, input_channels: int = 3, ndf: int = 64):
		super(full_conv_disc_64, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False) # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
		self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
		self.batch2 = nn.BatchNorm2d(ndf * 2)
		self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
		self.batch3 = nn.BatchNorm2d(ndf * 4)
		self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
		self.batch4 = nn.BatchNorm2d(ndf * 8)
		self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), .2)
		x = F.leaky_relu(self.batch2(self.conv2(x)), .2)
		x = F.leaky_relu(self.batch3(self.conv3(x)), .2)
		x = F.leaky_relu(self.batch4(self.conv4(x)), .2)
		return torch.sigmoid(self.conv5(x))
		
class full_conv_disc_28(nn.Module):
	def __init__(self, input_channels):
		super(full_conv_disc_28, self).__init__()
		ndf = 64
		self.conv1 = nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False) # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
		self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
		self.batch2 = nn.BatchNorm2d(ndf * 2)
		self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
		self.batch3 = nn.BatchNorm2d(ndf * 4)
		self.conv4 = nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False)
		self.batch4 = nn.BatchNorm2d(ndf * 4)
		self.conv5 = nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False)

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x))
		x = F.leaky_relu(self.batch2(self.conv2(x)), .2)
		x = F.leaky_relu(self.batch3(self.conv3(x)), .2)
		x = F.leaky_relu(self.batch4(self.conv4(x)), .2)
		return torch.sigmoid(self.conv5(x))

class full_conv_gen_28(nn.Module):
	def __init__(self, input_channels, output_channels: int = 1):
		super(full_conv_gen_28, self).__init__()
		ngf = 64
		self.conv1 = nn.ConvTranspose2d(input_channels, ngf * 8, 4, 1, 0, bias=False) # (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
		self.batch1 = nn.BatchNorm2d(ngf * 8)
		self.conv2 = nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False)
		self.batch2 = nn.BatchNorm2d(ngf * 4)
		self.conv3 = nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 2, bias=False)
		self.batch3 = nn.BatchNorm2d(ngf * 2)
		self.conv4 = nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False)
		self.batch4 = nn.BatchNorm2d(ngf)
		self.conv5 = nn.ConvTranspose2d( ngf, output_channels, 3, 1, 1, bias=False)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = F.relu(self.batch1(self.conv1(x)))
		x = F.relu(self.batch2(self.conv2(x)))
		x = F.relu(self.batch3(self.conv3(x)))
		x = F.relu(self.batch4(self.conv4(x)))
		return self.tanh(self.conv5(x))
		
class full_conv_gen_64_resid(nn.Module):
	def __init__(self, input_channels: int = 100, out_channels: int = 3, ngf: int = 64):
		super(full_conv_gen_64_resid, self).__init__()
		self.conv1 = nn.ConvTranspose2d(input_channels, ngf * 8, 4, 1, 0, bias=False)
		self.batch1 = nn.BatchNorm2d(ngf * 8)# state size. (ngf*8) x 4 x 4
		
		self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
		self.batch2 = nn.BatchNorm2d(ngf * 4)# state size. (ngf*4) x 8 x 8
		
		self.conv3 = nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False)
		self.batch3 = nn.BatchNorm2d(ngf * 2)# state size. (ngf*2) x 16 x 16
		
		self.conv4 = nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False)
		self.batch4 = nn.BatchNorm2d(ngf)# state size. (ngf) x 32 x 32
		
		self.conv5 = nn.ConvTranspose2d( ngf, out_channels, 4, 2, 1, bias=False)
		self.tanh = nn.Tanh()
		
		self.upsample = nn.Upsample(scale_factor = 2)
		self.channel_pool = channel_pool(2)
		
	def forward(self, x):
		x = F.relu(self.batch1(self.conv1(x)))
		res = x
		x = F.relu(self.batch2(self.conv2(x)))
		x += self.channel_pool(self.upsample(res))
		res = x
		x = F.relu(self.batch3(self.conv3(x)))
		x += self.channel_pool(self.upsample(res))
		res = x
		x = F.relu(self.batch4(self.conv4(x)))
		x += self.channel_pool(self.upsample(res))
		return self.tanh(self.conv5(x))
		
class full_conv_disc_64_resid(nn.Module):
	def __init__(self, input_channels: int = 3, ndf: int = 64):
		super(full_conv_disc_64_resid, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False) # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
		
		self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
		self.batch2 = nn.BatchNorm2d(ndf * 2)
		self.skip2 = nn.Conv2d(ndf, ndf * 2, 1, 2, bias=False)
		
		self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
		self.batch3 = nn.BatchNorm2d(ndf * 4)
		self.skip3 = nn.Conv2d(ndf * 2, ndf * 4, 1, 2, bias=False)
		
		self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
		self.batch4 = nn.BatchNorm2d(ndf * 8)
		self.skip4 = nn.Conv2d(ndf * 4, ndf * 8, 1, 2, bias=False)
		
		self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), .2)
		res = x
		x = F.leaky_relu(self.batch2(self.conv2(x)), .2)
		x += self.skip2(res)
		res = x
		x = F.leaky_relu(self.batch3(self.conv3(x)), .2)
		x += self.skip3(res)
		res = x
		x = F.leaky_relu(self.batch4(self.conv4(x)), .2)
		x += self.skip4(res)
		return torch.sigmoid(self.conv5(x))
		
class channel_pool(torch.nn.AvgPool1d):
	def forward(self, input):
		n, c, w, h = input.size()
		input = input.view(n,c,w*h).permute(0,2,1)
		pooled =  F.avg_pool1d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
		_, _, c = pooled.size()
		pooled = pooled.permute(0,2,1)
		return pooled.view(n, c, w, h)