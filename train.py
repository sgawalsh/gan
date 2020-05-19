import math
import numpy as np
import torch
import torch.nn as nn
import models
import data

from pdb import set_trace
from typing import List, Tuple

def train(input_shape, load_gen, load_disc, gen_true, loss = nn.BCELoss(), batch_size: int = 1, training_steps: int = 500, noise_length: int = 8, show_sample = True, save_models = True, gen_name: str = "gen", disc_name: str = "disc"):
	
	generator = load_gen(noise_length, input_shape)
	discriminator = load_disc(input_shape)
	
	generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
	discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
	
	for i in range(training_steps):
		noise = torch.randint(0, 2, size=(batch_size, noise_length)).float()
		generated_data = generator(noise)
		
		true_data = gen_true(input_shape, batch_size = batch_size)
		true_labels = torch.tensor([1] * batch_size).float() # true labels
		
		generator_optimizer.zero_grad()
		generator_discriminator_out = discriminator(generated_data)
		generator_loss = loss(generator_discriminator_out.view(batch_size), true_labels)
		generator_loss.backward() # call loss only on generator
		generator_optimizer.step()
		
		discriminator_optimizer.zero_grad()
		true_discriminator_out = discriminator(true_data)
		true_discriminator_loss = loss(true_discriminator_out.view(batch_size), true_labels)
		
		generator_discriminator_out = discriminator(generated_data.detach())
		#generator_discriminator_out = discriminator(generated_data)
		generator_discriminator_loss = loss(generator_discriminator_out.view(batch_size), torch.zeros(batch_size))
		discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
		discriminator_loss.backward()
		discriminator_optimizer.step()
		
	if show_sample:
		for i in range(10):
			print(generator(torch.randint(0, 2, size=(1, noise_length)).float()))
	
	#save models
	if save_models:
		torch.save(generator.state_dict(), "models/" + gen_name)
		torch.save(discriminator.state_dict(), "models/" + disc_name)
	return
	
def sample(modelClass, input_length: int, output_length: torch.Size, sample_num: int = 10, modelName: str = "gen"):
	model = modelClass(input_length, output_length)
	model.load_state_dict(torch.load("models/" + modelName))
	model.eval()
	for i in range(sample_num):
		print(model(torch.randint(0, 2, size=(1, input_length)).float()))
	return


#set_trace()
train(torch.Size([5]), models.sample_generator, models.sample_discriminator, data.generate_even_data, batch_size = 16)
sample(models.sample_generator, 8, torch.Size([5]))
print("done")