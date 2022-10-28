import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.nn.parameter import Parameter
from networks import ActorNetwork, DynamicsModel
from utils import ReplayBuffer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent:
	def __init__(self, state_dim, action_dim, max_action, alpha=0.0003, gamma=0.99,
				max_size=100_000, batch_size=64, ensemble_size=3):
		# init hyperparameters
		self.gamma = gamma
		self.batch_size = batch_size
		self.ensemble_size = 3
		self.planning_iters = 100

		# agent variables
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.ensemble_size = ensemble_size
		self.max_action = max_action

		# init networks
		self.actor = ActorNetwork(alpha, state_dim, action_dim, max_action)
		self.ensemble, self.memory = self.init_ensemble(max_size)

	def init_ensemble(self, max_size):
		ensemble, memory = [], []
		for i in range(self.ensemble_size):
			ensemble.append(DynamicsModel(self.state_dim, self.action_dim, str(i)))
			memory.append(ReplayBuffer(max_size//self.ensemble_size, self.state_dim, self.action_dim))
		return ensemble, memory

	def calc_disagreement(self, state, action):
		model_predictions = torch.zeros((self.ensemble_size, self.batch_size, self.state_dim))
		for i, model in enumerate(self.ensemble):
			mu, _ = model(state, action)
			model_predictions[i, ...] = mu # (B, S)
		disagreement = torch.var(model_predictions, dim=0).norm() # Frobenius norm of variance matrix
		return disagreement

	def store_transition(self, state, action, reward, state_, done):
		random_memory = random.choice(self.memory) # add transition to a some ensemble's memory
		random_memory.store_transition(state, action, reward, state_, done)

	def choose_action(self, state):
		state = torch.tensor([state]).to(device)
		actions = self.actor.sample_normal(state, reparameterize=False)
		return actions.cpu().detach().numpy()[0]

	def learn_policy(self):
		# TODO: make policy access the aggregate memory of all models
		if self.memory[0].mem_ctr < self.batch_size:
			return # don't learn until we can sample at least a full batch

		# Sample memory buffer uniformly
		states, actions, rewards, states_, done = self.memory[0].sample_buffer(self.batch_size)

		# Convert from numpy arrays to torch tensors for computation graph
		states = torch.tensor(states, dtype=torch.float).to(device)
		actions = torch.tensor(actions, dtype=torch.float).to(device)
		states_ = torch.tensor(states_, dtype=torch.float).to(device)
		rewards = torch.tensor(rewards, dtype=torch.float).to(device)
		done = torch.tensor(done, dtype=torch.float).to(device)

		# <---- ACTOR UPDATE ---->
		self.actor.optimizer.zero_grad()
		horizon = 1 # 10
		disagreement_loss = 0
		for _ in range(horizon):
			actions = self.actor.sample_normal(states, reparameterize=True)
			disagreement_loss -= self.calc_disagreement(states, actions)
			random_model = random.choice(self.ensemble)
			states = random_model(states, actions)
		disagreement_loss.backward()
		self.actor.optimizer.step()

	# TODO: each ensemble should learn on a different fold of data
	# Options:
	# 1) implement this at the replay buffer level (Easy)
	# 2) Other option is to sample different batches, but this doesn't work
	def learn_ensemble(self, ensemble_iters=100):
		for memory, model in zip(self.memory, self.ensemble):
			if memory.mem_ctr < self.batch_size:
				continue # don't learn until we can sample at least a full batch
			
			for epoch in range(ensemble_iters):
				# Sample memory buffer uniformly
				states, actions, rewards, states_, done = memory.sample_buffer(self.batch_size)

				# Convert from numpy arrays to torch tensors for computation graph
				states = torch.tensor(states, dtype=torch.float).to(device)
				actions = torch.tensor(actions, dtype=torch.float).to(device)
				states_ = torch.tensor(states_, dtype=torch.float).to(device)
				rewards = torch.tensor(rewards, dtype=torch.float).to(device)
				done = torch.tensor(done, dtype=torch.float).to(device)

				# <---- ENSEMBLE UPDATE ---->
				model.optimizer.zero_grad()
				next_states = model.sample_normal(states, actions, reparameterize=True)
				loss = F.mse_loss(states_, next_states)
				loss.backward()
				model.optimizer.step()

	def save_models(self):
		self.actor.save_checkpoint()
		for model in self.ensemble:
			model.save_checkpoint()

	def load_models(self):
		self.actor.load_checkpoint()
		for model in self.ensemble:
			model.load_checkpoint()
