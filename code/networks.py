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

reparam_noise = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DynamicsModel(nn.Module):
	'''
	Dynamics Model maps (state, action) -> state
	'''

	def __init__(self, state_dim, action_dim, name, fc1_dims=128, fc2_dims=128, chkpt_dir='../tmp'):
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
		self.to(device)

	def forward(self, state, action, debug=False):
		x = torch.cat([state, action], dim=-1) # (B, 23)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)

		mu = self.mu(x)
		sigma = torch.sigmoid(self.sigma(x)) # bound between 0 and 1
		sigma_clamp = torch.clamp(sigma, min=reparam_noise) # lower bound for sigma

		if debug:
			# mu, sigma: (B, 3)
			print(torch.cat([mu, sigma], dim=-1))

		return mu, sigma_clamp

	def sample_normal(self, state, action, reparameterize=True):
		mu, sigma = self.forward(state, action)
		probabilities = torch.distributions.Normal(mu, sigma)

		if reparameterize: # use the reparameterization trick
			next_state = probabilities.rsample() # scale and shift a standard normal -> gives differentiable sample
		else:
			next_state = probabilities.sample() # sample is non-differentiable

		return next_state

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.chkpt_file)
	
	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.chkpt_file))

class ActorNetwork(nn.Module):
	def __init__(self, alpha, state_dim, action_dim, max_action,
				 fc1_dims=128, fc2_dims=128, chkpt_dir='../tmp'):
		super(ActorNetwork, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim 
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.chkpt_dir = chkpt_dir
		self.chkpt_file = f"{chkpt_dir}/actor"
		self.max_action = max_action
		self.reparam_noise = 1e-6

		self.fc1 = nn.Linear(state_dim, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)

		self.mu = nn.Linear(fc2_dims, action_dim)
		self.sigma = nn.Linear(fc2_dims, action_dim)

		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.to(device)

	def forward(self, state):
		prob = self.fc1(state)
		prob = F.relu(prob)
		prob = self.fc2(prob)
		prob = F.relu(prob)

		mu = self.mu(prob)
		sigma = torch.sigmoid(self.sigma(prob)) # bound between 0 and 1
		sigma_clamp = torch.clamp(sigma, min=reparam_noise) # lower bound for sigma
		return mu, sigma_clamp, sigma

	def sample_normal(self, state, reparameterize=True):
		mu, sigma, before_sigma = self.forward(state)
		# if state.shape[0] == 1 and np.random.random() < 0.05:
		# 	print(f"Debug: mu={mu.item()}, before_sigma={before_sigma.item()}, after_sigma={sigma.item()}")
		probabilities = torch.distributions.Normal(mu, sigma)

		if reparameterize: # use the reparameterization trick
			actions = probabilities.rsample() # scale and shift a standard normal -> gives differentiable sample
		else:
			actions = probabilities.sample() # sample is non-differentiable

		action = torch.tanh(actions) * torch.tensor(self.max_action).to(device)
		return action

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.chkpt_file)
	
	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.chkpt_file))
