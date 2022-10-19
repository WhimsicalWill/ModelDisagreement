import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# TODO: for images, use CNN to embed images to low dimensional space
# class Embedding(nn.Module):

# Plan:
# Use FC network for dynamics model to begin with
# Use like 3 of them
# Use online planning with a planning horizon of 1 (we can expand horizon later)
# Don't overcomplicate things until the workflow works

class DynamicsModel(nn.Module):
	'''
	Dynamics Model maps (state, action) -> state
	'''

	def __init__(self, state_dim, action_dim, name, fc1_dims=128, fc2_dims=128, chkpt_dir='../tmp/dynamics'):
		super(DynamicsModel, self).__init__()
		self.state_dim = state_dim
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.action_dim = action_dim
		self.chkpt_file = f"{chkpt_dir}/dynamics_{name}"

		self.fc1 = nn.Linear(state_dim + action_dim, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)

		self.mu = nn.Linear(fc2_dims, state_dim)
		self.sigma = nn.Linear(fc2_dims, state_dim)

		self.optimizer = optim.Adam(self.parameters(), 0.0003)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state, action):
		# print(f"Shapes: {state.shape}, {action.shape}")
		x = torch.cat([state, action])
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)

		mu = self.mu(x)
		sigma = self.sigma(x)
		return mu, sigma

	def sample_normal(self, state, action, reparameterize=True):
		mu, sigma = self.forward(state, action)
		probabilities = torch.distributions.Normal(mu, sigma)

		if reparameterize: # use the reparameterization trick
			next_state = probabilities.rsample() # scale and shift a standard normal -> gives differentiable sample
		else:
			next_state = probabilities.sample() # sample is non-differentiable

		action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
		log_probs = probabilities.log_prob(actions)
		log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise) # TODO: what is this doing
		log_probs = log_probs.sum(1, keepdim=True) # idk if this is necessary (summing over only 1 col?)

		return action, log_probs