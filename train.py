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

def train(load_gen, load_disc, true_gen, loss = nn.BCELoss(), batch_size: int = 1, training_steps: int = 1000, noise_length: int = 100, show_sample = True, save_models = True, gen_name: str = '', disc_name: str = '', update: int = 50, show_plot = True, resize: int = 64, model_name: str = '', input_channels: int = 3):
	#fixed_noise = torch.randn(64, noise_length, 1, 1)
	
	generator = load_gen(noise_length, input_channels).to(device)
	discriminator = load_disc(input_channels).to(device)
	
	if not gen_name:
		generator.apply(weights_init)
	else:
		generator.load_state_dict(torch.load("models/" + gen_name))
		generator.eval()
	
	if not disc_name:
		discriminator.apply(weights_init)
	else:
		discriminator.load_state_dict(torch.load("models/" + disc_name))
		discriminator.eval()
	
	generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
	discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
	
	true_labels = torch.tensor([1] * batch_size).float().to(device) # true labels
	true_data_gen = true_gen(model_name, batch_size = batch_size, resize = resize)
	
	disc_loss_all, gen_loss_all, disc_loss_current, gen_loss_current = [],[],[],[]
	
	for i in range(training_steps):
		#train discriminator on real
		true_data = true_data_gen.__next__()
		#data.tensor_to_image(true_data.narrow_copy(0, 0, 1)) # show sample image
		
		#discriminator_optimizer.zero_grad()
		discriminator.zero_grad()
		true_discriminator_out = discriminator(true_data).view(-1) # pass true data to discrimintor
		true_discriminator_loss = loss(true_discriminator_out, true_labels) # calculate loss
		true_discriminator_loss.backward() # calculate gradients
		
		#train discriminator on fake
		noise = torch.randn(batch_size, noise_length, 1, 1, device=device) # get noise
		generated_data = generator(noise) # generate fake data
		generator_discriminator_out = discriminator(generated_data.detach()).view(-1) # pass fake data to discriminator
		generator_discriminator_loss = loss(generator_discriminator_out.view(batch_size), torch.zeros(batch_size).to(device)) # get loss
		generator_discriminator_loss.backward() # calculate gradients
		#discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
		#discriminator_loss.backward()
		discriminator_optimizer.step()
		
		noise = torch.randn(batch_size, noise_length, 1, 1, device = device)
		generated_data = generator(noise)
		
		# train generator
		generator_optimizer.zero_grad()
		generator_discriminator_out = discriminator(generated_data).view(-1)
		generator_loss = loss(generator_discriminator_out, true_labels).to(device) # generate loss, target true labels
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
		for i in range(20):
			data.tensor_to_image(generator(torch.randn(1, noise_length, 1, 1, device = device)).detach(), i + 1, "{0}_{1}_{2}".format(generator._get_name(), model_name, training_steps), show_image = False)
	
	#save models
	if save_models:
		torch.save(generator.state_dict(), "models/{0}_x_{1}_gen_{2}_{3}".format(generator._get_name(), discriminator._get_name(), model_name, training_steps))
		torch.save(discriminator.state_dict(), "models/{0}_x_{1}_disc_{2}_{3}".format(generator._get_name(), discriminator._get_name(), model_name, training_steps))
	return
	
def sample(modelClass, sample_num: int = 10, modelName: str = "gen", input_length: int = 100, channels: int = 3):
	model = modelClass(input_length, channels).to(device)
	model.load_state_dict(torch.load("models/" + modelName))
	model.eval()
	for i in range(sample_num):
		data.tensor_to_image(model(torch.randn(1, input_length, 1, 1, device = device)).detach())
	return

def preview_true(true_data_gen, samples: int = 5):
	for i in range(samples):
		data.tensor_to_image(true_data_gen.__next__())
#set_trace()
#train(torch.Size([5]), models.sample_generator, models.sample_discriminator, data.generate_even_data, batch_size = 16, training_steps = 1000)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train(models.full_conv_gen_64_resid, models.full_conv_disc_64, data.load_image_folder, batch_size = 16, training_steps = 10000, update = 50, model_name = 'dogs')

#train(models.full_conv_gen_64, models.full_conv_disc_64, data.load_image_folder, batch_size = 16, training_steps = 2000, update = 50, model_name = 'celeb', gen_name = "full_conv_gen_64_celeb_8000", disc_name = "full_conv_disc_64_celeb_8000")

#sample(models.full_conv_gen_64_resid, modelName = "full_conv_gen_64_resid_x_full_conv_disc_64_gen_flowers_8000", sample_num = 10)
#preview_true(data.load_image_folder("flowers"), 5)