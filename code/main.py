import gym
import sys
import getopt
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent_class import Agent, device
from utils import plot_learning_curve, render_games

def train(env_name):
	env = gym.make(env_name)
	s1, s2 = env.observation_space.shape[0], env.action_space.shape[0]
	print(f"ObsShape: {s1}, ActionShape: {s2}")
	agent = Agent(env.observation_space.shape[0], \
					env.action_space.shape[0], env.action_space.high[0])
	total_steps = 3e4
	ensemble_learn_interval = 1_000
	test_interval = total_steps // 10
	random_steps = 10_000
	best_score = env.reward_range[0] # init to smallest possible reward
	scores = []
	steps, episodes = 0, 0

	# Fill replay buffer with random transitions to train initial ensemble
	burn_in(env, agent, random_steps)

	# Update policy at each step, and update ensemble every K steps
	pbar = tqdm(total=total_steps)
	while steps < total_steps:
		done, testFlag = False, False
		observation = env.reset()
		score, prevSteps = 0, steps
		episodes += 1
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			agent.store_transition(observation, action, reward, observation_, done)
			if steps % ensemble_learn_interval == 0:
				agent.learn_ensemble()
			if steps % test_interval == 0:
				testFlag = True
			agent.learn_policy()
			score += reward
			steps += 1
			observation = observation_
		pbar.update(steps - prevSteps)
		scores.append(score)
		avg_score = np.mean(scores[-100:])

		if testFlag:
			test_agent(env, agent)
			testFlag = False

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()
		# print(f"Episode {episodes}, steps: {steps}, score: {score}, avg_score: {avg_score}")
	
	pbar.close()
	env.close()
	filename = f'{env_name}_{episodes}_games'
	figure_file = f'../plots/{filename}.png'
	plot_learning_curve(scores, figure_file)

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

def test_agent(env, agent):

	def get_ensemble_disagreement(state, action):
		state = torch.tensor([state], dtype=torch.float).to(device)
		action = torch.tensor([action], dtype=torch.float).to(device)
		return agent.calc_disagreement(state, action).item()

	rewards_i, rewards = [], []
	test_eps = 20
	for ep in range(test_eps):
		done, score_i, score = False, 0, 0
		state = env.reset()
		while not done:
			action = agent.choose_action(state)
			state_, reward, done, info = env.step(action)
			score_i += get_ensemble_disagreement(state, action)
			score += reward
			state = state_
		rewards_i.append(score_i)
		rewards.append(score)
	avg_i, avg = sum(rewards_i)/test_eps, sum(rewards)/test_eps
	print(f"I: {avg_i}, E: {avg}")


if __name__ == '__main__':
	arg_env_name = 'Pendulum-v1'
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