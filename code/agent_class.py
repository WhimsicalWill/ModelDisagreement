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
		self.memory = ReplayBuffer(max_size, state_dim, action_dim)
		self.ensemble = self.init_model_ensemble()

	def init_model_ensemble(self):
		ensemble = []
		for i in range(self.ensemble_size):
			ensemble.append(DynamicsModel(self.state_dim, self.action_dim, str(i)))
		return ensemble

	def calc_disagreement(self, state, action):
		model_predictions = torch.zeros((self.ensemble_size, self.batch_size, self.state_dim))
		for i, model in enumerate(self.ensemble):
			mu, _ = model(state, action)
			model_predictions[i, ...] = mu # (B, S)
		disagreement = torch.var(model_predictions, dim=0).norm() # Frobenius norm of variance matrix
		return disagreement

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def choose_action(self, state):
		state = torch.tensor([state]).to(device)
		actions = self.actor.sample_normal(state, reparameterize=False)
		return actions.cpu().detach().numpy()[0]

	def learn_policy(self):
		if self.memory.mem_ctr < self.batch_size:
			return # don't learn until we can sample at least a full batch

		# Sample memory buffer uniformly
		states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

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
			disagreement_loss += self.calc_disagreement(states, actions)
			random_model = random.choice(self.ensemble)
			states = random_model(states, actions)
		disagreement_loss.backward()
		self.actor.optimizer.step()

	def learn_ensemble(self):
		if self.memory.mem_ctr < self.batch_size:
			return # don't learn until we can sample at least a full batch

		# TODO: create PyTorch data loader so we can sample data uniformly
		# and iterate over (possibly multiple?) epochs of data to train models
		
		ensemble_iters = 100
		# For now, update the ensemble for a fixed number of iterations
		for epoch in range(ensemble_iters):
			# Sample memory buffer uniformly
			states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

			# Convert from numpy arrays to torch tensors for computation graph
			states = torch.tensor(states, dtype=torch.float).to(device)
			actions = torch.tensor(actions, dtype=torch.float).to(device)
			states_ = torch.tensor(states_, dtype=torch.float).to(device)
			rewards = torch.tensor(rewards, dtype=torch.float).to(device)
			done = torch.tensor(done, dtype=torch.float).to(device)

			# TODO: maybe implement dropout so different models see different subsets of data
			# <---- ENSEMBLE UPDATE ---->
			for model in self.ensemble:
				model.optimizer.zero_grad()
				next_states = model.sample_normal(states, actions, reparameterize=True) # model(states, actions)
				loss = F.mse_loss(states_, next_states)
				loss.backward()
				model.optimizer.step()
			# print(f"Ensemble learning, epoch: {epoch} finished, last loss: {loss}")

	def save_models(self):
		self.actor.save_checkpoint()
		for model in self.ensemble:
			model.save_checkpoint()

	def load_models(self):
		self.actor.load_checkpoint()
		for model in self.ensemble:
			model.load_checkpoint()
