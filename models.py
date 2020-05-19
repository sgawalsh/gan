import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class sample_generator(nn.Module):

	def __init__(self, input: int, output: torch.Size):
		super(sample_generator, self).__init__()
		self.dense = nn.Linear(input, output[0])
		self.activation = nn.Sigmoid()

	def forward(self, x):
		return self.activation(self.dense(x))


class sample_discriminator(nn.Module):
	def __init__(self, input: torch.Size):
		super(sample_discriminator, self).__init__()
		self.dense = nn.Linear(input[0], 1)
		self.activation = nn.Sigmoid()

	def forward(self, x):
		return self.activation(self.dense(x))

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