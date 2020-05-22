import math
import numpy as np
import torch
import torch.nn as nn
import models
import data
import matplotlib.pyplot as plt

from pdb import set_trace
from typing import List, Tuple

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

def train(load_gen, load_disc, gen_true, loss = nn.BCELoss(), batch_size: int = 1, training_steps: int = 1000, noise_length: int = 100, show_sample = False, save_models = True, gen_name: str = "gen", disc_name: str = "disc", new_gen = True, new_disc = True, update: int = 50, show_plot = True, resize: int = 64):
	#fixed_noise = torch.randn(64, noise_length, 1, 1)
	
	generator = load_gen(noise_length, 3)
	discriminator = load_disc(3)
	
	if new_gen:
		generator.apply(weights_init)
	else:
		generator.load_state_dict(torch.load("models/" + gen_name))
		generator.eval()
	
	if new_disc:
		discriminator.apply(weights_init)
	else:
		discriminator.load_state_dict(torch.load("models/" + disc_name))
	
	generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
	discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
	
	true_data_gen = gen_true(batch_size = batch_size, resize = resize)
	set_trace()
	true_labels = torch.tensor([1] * batch_size).float() # true labels
	
	disc_loss_all, gen_loss_all, disc_loss_current, gen_loss_current = [],[],[],[]
	
	for i in range(training_steps):
		
		#train discriminator on real
		true_data = true_data_gen.__next__()
		data.tensor_to_image(true_data.narrow_copy(0, 0, 1))
		
		set_trace()
		
		#discriminator_optimizer.zero_grad()
		discriminator.zero_grad()
		true_discriminator_out = discriminator(true_data).view(-1) # pass true data to discrimintor
		true_discriminator_loss = loss(true_discriminator_out, true_labels) # calculate loss
		true_discriminator_loss.backward() # calculate gradients
		
		#train discriminator on fake
		noise = torch.randn(batch_size, noise_length, 1, 1) # get noise
		generated_data = generator(noise) # generate fake data
		generator_discriminator_out = discriminator(generated_data.detach()).view(-1) # pass fake data to discriminator
		generator_discriminator_loss = loss(generator_discriminator_out.view(batch_size), torch.zeros(batch_size)) # get loss
		generator_discriminator_loss.backward() # calculate gradients
		#discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
		#discriminator_loss.backward()
		discriminator_optimizer.step()
		
		noise = torch.randn(batch_size, noise_length, 1, 1)
		generated_data = generator(noise)
		
		# train generator
		generator_optimizer.zero_grad()
		generator_discriminator_out = discriminator(generated_data).view(-1)
		generator_loss = loss(generator_discriminator_out, true_labels) # generate loss, target true labels
		generator_loss.backward() # calculate gradients
		generator_optimizer.step()
		
		#track loss
		disc_loss_current.append(generator_discriminator_loss + true_discriminator_loss)
		gen_loss_current.append(generator_loss)
		
		if not (i + 1) % update and i:
			d_mean = sum(disc_loss_current) / update
			g_mean = sum(gen_loss_current) / update
			print("Step {0}/{1} Discriminator Loss : {2} Generator Loss : {3}".format(i + 1, training_steps, round(d_mean.item(), 2), round(g_mean.item(), 2)))
			
			disc_loss_all.append(d_mean)
			gen_loss_all.append(g_mean)
			disc_loss_current, gen_loss_current = [],[]
			
	if show_plot:
		plt.figure(figsize=(10,5))
		plt.subplot(211)
		plt.title("Generator Loss")
		plt.plot(gen_loss_all)
		plt.xlabel("Iterations")
		plt.ylabel("Loss")
		plt.subplot(212)
		plt.title("Discriminator Loss")
		plt.plot(disc_loss_all)
		plt.xlabel("Iterations")
		plt.ylabel("Loss")
		plt.show()
		
	if show_sample:
		for i in range(3):
			#print(generator(torch.randint(0, 2, size=(1, noise_length)).float()))
			img = generator(torch.randn(1, noise_length, 1, 1))
			img = img.view(img.shape[2], img.shape[3])
			plt.imshow(img.detach())
			plt.show()
			
	
	#save models
	if save_models:
		torch.save(generator.state_dict(), "models/" + gen_name)
		torch.save(discriminator.state_dict(), "models/" + disc_name)
	return
	
def sample(modelClass, input_length: int, sample_num: int = 10, modelName: str = "gen"):
	model = modelClass(input_length, 3)
	model.load_state_dict(torch.load("models/" + modelName))
	model.eval()
	set_trace()
	for i in range(sample_num):
		img = model(torch.randn(1, input_length, 1, 1))
		img = img.view(img.shape[2], img.shape[3])
		plt.imshow(img.detach())
		plt.show()
	return

def preview_true(gen_true, samples: int = 5):
	true_data_gen = gen_true(3)
	for i in range(samples):
		data.tensor_to_image(true_data_gen.__next__())
#set_trace()
#train(torch.Size([5]), models.sample_generator, models.sample_discriminator, data.generate_even_data, batch_size = 16, training_steps = 1000)
train(models.full_conv_gen_64, models.full_conv_disc_64, data.celeb, batch_size = 16, training_steps = 100, update = 50)

#sample(models.full_conv_gen, 100, modelName = "gen", sample_num = 5)
#preview_true(data.fashion, 5)