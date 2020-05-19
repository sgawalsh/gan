import math
import numpy as np
import torch
import torch.nn as nn

from pdb import set_trace

class sample_generator(nn.Module):

	def __init__(self, input: int, output: torch.Size):
		super(sample_generator, self).__init__()
		self.dense_layer = nn.Linear(input, output[0])
		self.activation = nn.Sigmoid()

	def forward(self, x):
		return self.activation(self.dense_layer(x))


class sample_discriminator(nn.Module):
	def __init__(self, input: torch.Size):
		super(sample_discriminator, self).__init__()
		self.dense = nn.Linear(input[0], 1)
		self.activation = nn.Sigmoid()

	def forward(self, x):
		return self.activation(self.dense(x))
