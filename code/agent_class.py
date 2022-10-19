import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.parameter import Parameter
from networks import DynamicsModel
from utils import ReplayBuffer

class Agent:
	def __init__(self, state_dim, action_dim, max_action, gamma=0.99,
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
		self.memory = ReplayBuffer(max_size, state_dim, action_dim)
		self.ensemble = self.init_model_ensemble()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	def init_model_ensemble(self):
		ensemble = []
		for i in range(self.ensemble_size):
			ensemble.append(DynamicsModel(self.state_dim, self.action_dim, str(i)))
		return ensemble

	def calc_disagreement(self, state, action):
		model_predictions = torch.zeros((self.ensemble_size, self.state_dim))
		for i, model in enumerate(self.ensemble):
			mu, sigma = model(state, action)
			model_predictions[i] = mu
		disagreement = torch.var(model_predictions, dim=0).norm()
		return disagreement

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def choose_action(self, state):
		# initialize action uniformly between valid range
		state = torch.tensor([state]).to(self.device)
		print(f"State shape: {state.shape}")
		action = torch.rand((1, self.action_dim)).to(self.device)
		action = self.max_action * (action * 2 - 1)
		action.requires_grad_()

		optimizer = optim.Adam([action], lr=0.01)
		for i in range(self.planning_iters):
			optimizer.zero_grad()
			disagreement_loss = -self.calc_disagreement(state, action)
			# print(f"Optimizing action step {i}, disagreement={disagreement_loss.item()} action={action}")
			disagreement_loss.backward()
			optimizer.step()

		return action.cpu().detach().numpy()

	def learn(self):
		if self.memory.mem_ctr < self.batch_size:
			return # don't learn until we can sample at least a full batch

		# Sample memory buffer uniformly
		states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

		# Convert from numpy arrays to torch tensors for computation graph
		states = torch.tensor([states], dtype=torch.float).to(self.actor.device)
		actions = torch.tensor([actions], dtype=torch.float).to(self.actor.device)
		states_ = torch.tensor([states_], dtype=torch.float).to(self.actor.device)
		rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
		done = torch.tensor(done, dtype=torch.float).to(self.actor.device)

		# Do one gradient step for all models in ensemble
		for model in self.ensemble:
			model.optimizer.zero_grad()
			next_states = model(states, actions)
			loss = F.mse_loss(states_, next_states)
			loss.backward()
			model.step()

	def save_models(self):
		for model in self.ensemble:
			model.save_checkpoint()

	def load_models(self):
		for model in self.ensemble:
			model.load_checkpoint()
