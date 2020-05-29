import math
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

import urllib, io, os
from PIL import Image

from typing import List, Tuple
from pdb import set_trace
from google_images_download import google_images_download

import requests, os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def image_from_urls(filepath: str, encoding: str = "utf8", threshold = .2, filter = True):
	with open(filepath, 'r', encoding = encoding) as f:
		urls = f.read()
		f.close()
	
	urls = urls.split()
	
	loc_data = "./data/{0}/".format(filepath.split('.')[0])
	try:
		os.makedirs(loc_data)
	except:
		pass
		
	for i, url in enumerate(urls):
		try:
			response = urllib.request.urlopen(url)
			if response.status == 200:
				im = Image.open(io.BytesIO(response.read()))
				height, width = im.size[0], im.size[1]
				if (height > 256 and width > 256 and width > (1 - threshold) * height and width < (1 + threshold) * height) or not filter:
					im.save(loc_data + 'image{:05.0f}.jpg'.format(i), format = 'JPEG')
					
		except Exception as e:
			print("\n{} {}".format(e,url))
			pass


def google_images(keyword_str: str, out_dir: str = './data', limit: int = 5): # broken library :(
	response = google_images_download.googleimagesdownload()
	
	arguments = {"keywords":keyword_str,"limit":limit,"print_urls":True, "output_directory" : out_dir}   #creating list of arguments
	paths = response.download(arguments)   #passing the arguments to the function
	print(paths)   #printing absolute paths of the downloaded images

def tensor_to_image_plt(img, num: int = 0):
	img = np.squeeze(img.cpu())
	plt.imshow(img.permute(1,2,0))
	plt.show()
	if num:
		plt.savefig("sample_images/{0}.png".format(str(num)))
		
def tensor_to_image(t, num: int = 0, model_name: str = '', show_image = True):
	im = torchvision.transforms.ToPILImage()(t.squeeze().cpu()).convert("RGB")
	if show_image:
		im.show()
	if num:
		im.save("sample_images/{0}_{1}.jpg".format(model_name, str(num)), 'JPEG')

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
	
def mnist(batch_size: int = 1, resize: int = 28):
	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST(
			root = './data',
			train = True,
			download = True,
			transform = torchvision.transforms.Compose([
				torchvision.transforms.Resize(resize),
				torchvision.transforms.CenterCrop(resize),
				torchvision.transforms.ToTensor()
			])
		),
	batch_size=batch_size, shuffle=True)
	
	while True:
		ex = enumerate(train_loader)
		_, data = next(ex)
		yield data[0].to(device)
		
def fashion(batch_size: int = 1, resize: int = 28):
	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.FashionMNIST(
			root = './data',
			train = True,
			download = True,
			transform = torchvision.transforms.Compose([
				torchvision.transforms.Resize(resize),
				torchvision.transforms.CenterCrop(resize),
				torchvision.transforms.ToTensor()
			])
		),
	batch_size=batch_size, shuffle=True)
	
	while True:
		_, data = next(iter(train_loader))
		yield data[0].to(device)

def load_image_folder(folder_name: str, batch_size: int = 1, resize: int = 64):
	train_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
		root = './data/{0}'.format(folder_name),
		transform = torchvision.transforms.Compose([
			torchvision.transforms.Resize(resize),
			torchvision.transforms.CenterCrop(resize),
			torchvision.transforms.ToTensor(),
			#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
	), batch_size=batch_size, shuffle=True)
	while True:
		try:
			data, _ = next(iter(train_loader))
			yield data.to(device)
		except GeneratorExit:
			return
		
def general_loader(dataset, batch_size: int = 1):
	train_loader = torch.utils.data.DataLoader(
		dataset(
			root = data,
			train = True,
			download = True,
			transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor()                                 
			])
		),
	batch_size=batch_size, shuffle=True)
	
	while True:
		ex = enumerate(train_loader)
		_, data = next(ex)
		yield data[0].to(device)
		
def delete_broken(folder_path):
	for file_name in os.listdir(folder_path):
		try:
			Image.open(folder_path + "/" + file_name)
			continue
		except:
			os.remove(folder_path + "/" + file_name)
	
#image_from_urls("beverages.txt", filter = False)
#delete_broken("E:/New folder/Other/Programming/gan/data/beverages/beverages")