import math
import numpy as np
import torch
import torchvision

from typing import List, Tuple
from pdb import set_trace

def create_binary_list_from_int(number: int) -> List[int]:
	if number < 0 or type(number) is not int:
		raise ValueError("Only Positive integers are allowed")

	return [int(x) for x in list(bin(number))[2:]]

def generate_even_data(outShape, batch_size: int = 16) -> Tuple[List[int], List[List[int]]]:
	# Get the number of binary places needed to represent the maximum number
	max_length = outShape[0]

	while True:
		# Sample batch_size number of integers in range 0-max_int
		sampled_integers = np.random.randint(0, int(2 ** max_length / 2), batch_size)

		# Generate a list of binary numbers for training.
		data = [create_binary_list_from_int(int(x * 2)) for x in sampled_integers]
		data = [([0] * (max_length - len(x))) + x for x in data]

		yield torch.tensor(data).float()
	
def mnist(batch_size: int = 1):
	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST(
			root = './data',
			train = True,
			download = True,
			transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor()                                 
			])
		),
	batch_size=batch_size, shuffle=True)
	
	print(train_loader)
	while True:
		ex = enumerate(train_loader)
		_, data = next(ex)
		yield data