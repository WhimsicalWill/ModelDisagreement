import gym
import sys
import getopt
import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from hotstarts import HotStarts
from collect import collect_random_hot_starts
from tqdm import tqdm
from agent_class import Agent, device
from utils import *

def train(env_name):
	env = gym.make(env_name)

	# number of hot starts to render (must be a square)
	num_hot_starts = 4

	# setup HotStarts env wrapper
	saved_hot_starts_dir = 'data'
	env = HotStarts(env, saved_hot_starts_dir, num_hot_starts)
	collect_random_hot_starts(env, num_hot_starts)

	dim1, dim2 = env.observation_space.shape[0], env.action_space.shape[0]
	action_high = env.action_space.high[0]
	print(f"ObsShape: {dim1}, ActionShape: {dim2}, ActionHigh: {action_high}")
	agent = Agent(dim1, dim2, action_high)
	num_train_eps = 400
	ensemble_learn_interval = 1 # 20 * 200 = 4000 steps per update 
	burn_in_steps = 10_000
	test_rewards_i, test_rewards = [], []


	# Initialize linear schedule for the horizon parameter
	horizon_schedule = f"linear(1, 10, {num_train_eps})"

	# Fill replay buffer with random transitions to train initial ensemble
	burn_in(env, agent, burn_in_steps)

	# Update policy at each step, and update ensemble every K steps
	pbar = tqdm(total=num_train_eps, position=0, leave=True)
	for ep in range(num_train_eps):
		if ep % ensemble_learn_interval == 0:
			print(f"Learning ensemble, Episode={ep}")
			debug_agent(env, agent)
			agent.learn_ensemble(100)
		if ep % (num_train_eps // 50) == 0:
			env.visualize_hot_starts(agent.choose_action)
		done = False
		observation = env.reset()
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			agent.store_transition(observation, action, reward, observation_, done)
			horizon = int(linear_schedule(horizon_schedule, ep))
			agent.learn_policy(horizon)
			observation = observation_
		print(f"Episode: {ep}, action: {action}")
		pbar.update(1)
	pbar.close()
	env.close()

def burn_in(env, agent, total_steps):
	print("Collecting random experience")
	steps = 0
	while steps < total_steps:
		done = False
		observation = env.reset()
		while not done:
			action = env.action_space.sample()
			observation_, reward, done, info = env.step(action)
			agent.store_transition(observation, action, reward, observation_, done)
			steps += 1
			observation = observation_
	print("Finished collecting random experience")

def debug_agent(env, agent):
	agent_policy = lambda x: agent.choose_action(x)
	random_policy = lambda x: env.action_space.sample()
	r_i, r = test_agent(env, agent, agent_policy)
	_, _, = test_agent(env, agent, random_policy)
	agent.debug_model()

def test_agent(env, agent, policy):

	def get_ensemble_disagreement(state, action):
		state = torch.tensor([state], dtype=torch.float).to(device)
		action = torch.tensor([action], dtype=torch.float).to(device)
		return agent.calc_disagreement(state, action).item()

	rewards_i, rewards = [], []
	test_eps = 1
	for ep in range(test_eps):
		done, score_i, score = False, 0, 0
		state = env.reset()
		while not done:
			action = policy(state)
			state_, reward, done, info = env.step(action)
			score_i += get_ensemble_disagreement(state, action)
			score += reward
			state = state_
		rewards_i.append(score_i)
		rewards.append(score)
	avg_i, avg = sum(rewards_i)/test_eps, sum(rewards)/test_eps
	print(f"I: {avg_i}, E: {avg}")
	return avg_i, avg

def test_video(agent, env, env_name, ep):
	save_path = f"../videos/video-{env_name}-{ep}"
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	# To create video
	env = gym.wrappers.Monitor(env, save_path, force=True)
	state, done = env.reset(), False
	action_num = 0
	while not done:
		env.render()
		action = agent.choose_action(state)
		print(f"Action num {action_num}: {action}")
		action_num += 1
		next_state, reward, done, info = env.step(action)
		state = next_state
	env.close()


if __name__ == '__main__':
	arg_env_name = 'HalfCheetah-v3'
	arg_render = False
	arg_help = f"{sys.argv[0]} -e <env_name> | use -r to render games from saved policy"

	try:
		opts, args = getopt.getopt(sys.argv[1:], "hre:", ["help", "render", "env_name="])
	except:
		print(arg_help)
		sys.exit(2)

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(arg_help)
			sys.exit(2)
		elif opt in ("-e", "--env_name"):
			arg_env_name = arg
		elif opt in ("-r", "--render"):
			arg_render = True
	
	if arg_render:
		render_games(arg_env_name)
	else:
		train(arg_env_name)