import numpy as np
import os
import random 
import copy
import math
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from match3Env import Match3Env
from config import *
import time
from non_model import random_baseline, greedy_baseline
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Q_network(nn.Module):
	def __init__(self, n_action):
		super(Q_network, self).__init__()
		self.network = nn.Sequential(
			nn.Conv2d(in_channels=NUMOFCUBE, out_channels=30, kernel_size=4),
			nn.BatchNorm2d(30),
			nn.SELU(),
			nn.Conv2d(in_channels=30, out_channels=10, kernel_size=3),
			nn.ReLU(),
			nn.Dropout(p=.2),
			nn.Flatten(),
			nn.Linear(((((LEVEL - 4) + 1) - 3) + 1) ** 2 * 10, 128),   
			nn.Sigmoid(),
			nn.Linear(128, 1024),
			nn.Dropout(p=.1),
			nn.Linear(1024, 36),
			nn.ReLU(),
			nn.Linear(36, n_action),
		)

	def forward(self, x):
		return self.network(x)

def TO_ONE_HOT(x):
	x_one_hot = (np.arange(NUMOFCUBE) == x[..., None] - 1).astype(int)
	return np.transpose(x_one_hot, (2, 0, 1))

class Sample_data:
	def __init__(self, state, action, reward, next_state):
		self.state = state
		self.action = action
		self.reward = reward
		self.next_state = next_state

	def data(self):
		return self.state, self.action, self.reward, self.next_state

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Buffer:
	def __init__(self):
		self._buffer = []
		self.current_size = 0
		self.MAX_BUFFER_SIZE = BUFFER_CAPACITY

	def store(self, data:Sample_data):
		if(self.current_size < self.MAX_BUFFER_SIZE):
			self.current_size += 1
		else:
			self._buffer.pop(random.randint(0, int(self.current_size * BUFFER_UPDATE_FRACTION)))
		self._buffer.append(data)
		return self

	def sample(self, batch_size):
		idx = np.random.choice(np.arange(len(self._buffer)), size=BATCH_SIZE)
		return [self._buffer[ii] for ii in idx]
		# return self._buffer[random.randint(int(self.current_size * (1 - BUFFER_ACCESS_FRACTION)), self.current_size - 1)]

class ReplayBuffer(object):
	def __init__(self):
		self.capacity = BUFFER_CAPACITY
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		# cyclic buffer
		self.position = (self.position + 1) % self.capacity

	# selecting a random batch of transitions for training
	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

def select_action_epsilon_greedy(network, step, state, n_action):
	sample = random.random()
	if SELECT_ACTION_METHOD == 1:
		eps_threshold = 1 - pow(0.99999, step / EPS_DECAY) * EPSILON_GREEDY_INIT
		action_selected = -1
		if sample > eps_threshold: #rand
			action_selected = random.randint(0, n_action - 1)
		else:
			action_selected = network.forward(state)
			action_selected = torch.argmax(action_selected)
		return action_selected

	elif SELECT_ACTION_METHOD == 2:
		eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (step + 1) / EPS_DECAY)
		if sample > eps_threshold:
			with torch.no_grad():
				# t.max(1) will return largest column value of each row.
				# second column on max result is index of where max element was
				# found, so we pick action with the larger expected reward.
				return network(state).max(1)[1].view(1, 1)
		else:
			return torch.tensor([[random.randrange(n_action)]], device=device)
##selected actions can be matched
def select_action_epsilon_greedy_legal_action(network, step, state, n_action,legal_actions):
	sample = random.random()
	action_selected=0
	if SELECT_ACTION_METHOD == 1:
		action_selected_N = network.forward(state)
		action_selected_N = torch.argsort(action_selected_N)

		eps_threshold = 1 - pow(0.99999, step / EPS_DECAY) * EPSILON_GREEDY_INIT
		if sample > eps_threshold: #rand
			action_selected = random.randint(0, n_action - 1)
		else:
			# action_selected=action_selected_N
			for action in action_selected_N[0]:
				if action in legal_actions:
					action_selected=action
					break
				else:
					continue
		return action_selected

	# elif SELECT_ACTION_METHOD == 2:
	# 	eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (step + 1) / EPS_DECAY)
	# 	if sample > eps_threshold:
	# 		with torch.no_grad():
	# 			# t.max(1) will return largest column value of each row.
	# 			# second column on max result is index of where max element was
	# 			# found, so we pick action with the larger expected reward.
	# 			return network(state).max(1)[1].view(1, 1)
	# 	else:
	# 		return torch.tensor([[random.randrange(n_action)]], device=device)

def train():

	checkpoint_path = os.path.join(DIRPATH, "checkpoint.txt")
	if not os.path.exists(DIRPATH):
		print("Creating directory")
		os.mkdir(DIRPATH)
	if not os.path.exists(checkpoint_path):
		f = open(checkpoint_path, mode="w+")
		f.write("1")
		f.close()
	f = open(checkpoint_path, mode='r')
	checkpoint = f.read()
	checkpoint = int(checkpoint)
	f.close()

	print(f'Using cuda: {torch.cuda.is_available()}')
	begin_time = time.time()
	print(f'Start training at {time.ctime()}')

	env = Match3Env()
	env.reset()
	replay_buffer = ReplayBuffer()
	n_action = env.action_space.n
	policy_network = Q_network(n_action).to(device=device)
	target_network = copy.deepcopy(policy_network)

	try:
		policy_network.load_state_dict(torch.load(f=os.path.join(DIRPATH, f'checkpoint_{checkpoint - 1}.pt')))
	except:
		print("New network")

	optimizer = optim.RMSprop(policy_network.parameters())
	gg = False
	update_counter=1
	running_loss = 0
	training_counter=0
	# for i_episode in range(N_EPISODES):
	i_episode=0
	while True:

		env.reset()
		env.check_legal_step()
		current_state = env.get_board()
		current_state = TO_ONE_HOT(current_state)
		current_state = torch.tensor([current_state], device=device, dtype=torch.float)
		
		for step_counter in range(MAX_STEPS):
			action_selected = select_action_epsilon_greedy_legal_action(policy_network, step_counter, current_state, n_action,env.legal_action_index)
			next_state, reward, gg, info = env.step(action_selected)
			env.check_legal_step()
			next_state = TO_ONE_HOT(next_state)
			action_selected = torch.tensor([[action_selected]], device=device)
			next_state = torch.tensor([next_state], device=device, dtype=torch.float)
			reward = torch.tensor([reward], device=device)

			if gg:
				next_state = None
			
			replay_buffer.push(current_state, action_selected, next_state, reward)
			current_state = next_state
			
			# optimize_model
			if len(replay_buffer) > BATCH_SIZE and training_counter %128 ==0:
				transitions = replay_buffer.sample(BATCH_SIZE)
				batch = Transition(*zip(*transitions))
				non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
				state_batch = torch.cat(batch.state)
				action_batch = torch.cat(batch.action)
				# next_state_batch = torch.cat(batch.next_state)
				non_final_next_state_batch = torch.cat([s for s in batch.next_state if s is not None])
				reward_batch = torch.cat(batch.reward)

				# print('transitions.shape', np.shape(transitions))
				# print('batch.shape', np.shape(batch))
				# print('state_batch.shape', np.shape(state_batch))
				# print('action_batch.shape', np.shape(action_batch))
				# print('next_state_batch.shape', np.shape(next_state_batch))
				# print('reward_batch.shape', np.shape(reward_batch))

				td_estimate = policy_network(state_batch).gather(1, action_batch)
				# DQN
				# td_target = target_network(next_state_batch).max(1)[0].detach()
				# DDQN
				# td_target = target_network(next_state_batch).gather(1, torch.argmax(policy_network(next_state_batch), dim=1).unsqueeze(1)).detach()
				td_target = torch.zeros(BATCH_SIZE, device=device).unsqueeze(1)
				td_target[non_final_mask] = target_network(non_final_next_state_batch).gather(1, torch.argmax(policy_network(non_final_next_state_batch), dim=1).unsqueeze(1)).detach()
				td_target = (td_target * GAMMA) + reward_batch.unsqueeze(1)
				# Compute Huber loss

				loss = F.smooth_l1_loss(td_estimate, td_target)
				# Optimize the model
				optimizer.zero_grad()
				loss.backward()

				for param in policy_network.parameters():
					param.grad.data.clamp_(-1, 1)

				optimizer.step()
				running_loss += loss.item()
				update_counter+=1
			training_counter +=1
			# if(step_counter%CHECK_LEGAL_STEPS==0):
			# 	gg=env.check_legal_step()
			# print(f'train {training_counter} steps, update {update_counter} times')
			if RENDER:
				env.render()
			if PRINT_STEP:
				print(f'step:{update_counter}')
			if update_counter % STEP_UPDATE_NETWORK == 0:
				target_network.load_state_dict(policy_network.state_dict())
				torch.save(obj=target_network.state_dict(), f=os.path.join(DIRPATH, f'checkpoint_{checkpoint}.pt'))
				checkpoint = str(int(checkpoint) + 1)
				f = open(checkpoint_path, mode='w')
				f.write(checkpoint)
				f.close()
				print(f'Model saved to {DIRPATH}checkpoint_{checkpoint}.pt')
				update_counter+=1
			if gg:
				break

		if i_episode % EPISODE_DISPLAY == 0:
			print(f'Iterate {step_counter} steps\nUsing {int(time.time() - begin_time)} seconds')
			print(f'Current reward: {reward}')
			print(f'Real Total reward: {env.game.score}')
			# print('loss', running_loss)
			running_loss = 0


		# print(target_network.state_dict())
	target_network.load_state_dict(policy_network.state_dict())
	torch.save(obj=target_network.state_dict(), f=os.path.join(DIRPATH, f'checkpoint_{checkpoint}.pt'))
	checkpoint = str(int(checkpoint) + 1)
	f = open(checkpoint_path, mode='w')
	f.write(checkpoint)
	f.close()
	print(f'Model saved to {DIRPATH}checkpoint_{checkpoint}.pt')

def test():
	rl,gd,rd=0,0,0
	ary_gd=[]
	ary_rl=[]
	ary_rd=[]
	test_episodes=1000
	for _ in range(test_episodes):
		env = Match3Env()
		env.reset()
		env.check_legal_step()
		env_random = copy.deepcopy(env)
		random_model = random_baseline(env_random)


		env_greedy = copy.deepcopy(env)
		greedy_model = greedy_baseline(env_greedy)

		n_action = env.action_space.n
		test_network = Q_network(n_action).to(device=device)
		f = open(f'{DIRPATH}checkpoint.txt',mode = 'r')
		checkpoint=int(f.read())-1
		checkpoint_path = os.path.join(DIRPATH, f'checkpoint_{checkpoint}.pt')
		
		if os.path.exists(checkpoint_path):
			test_network.load_state_dict(torch.load(checkpoint_path))
			current_state = env.get_board()
			current_state = TO_ONE_HOT(current_state)
			current_state = torch.tensor([current_state], device=device, dtype=torch.float)
			total_reward = 0
			while True:
				g1,g2,g3=False,False,False

				random_model.execute()
				if random_model.game_over():
					g1=True
				greedy_model.execute()
				if greedy_model.game_over():
					g2=True
				# action_selected = test_network(current_state).max(1)[1].cpu().detach().numpy()[0]
				# action_selected = torch.tensor(action_selected, device=device)
				
				action_selected_N = torch.argsort(test_network.forward(current_state))
				action_selected = torch.argmax(test_network.forward(current_state))
				# print('action_selected_N', action_selected_N)
				for action in action_selected_N[0]:
					# print('action', action)
					# print(env.legal_action_index)
					if action in env.legal_action_index:
						
						action_selected=action
						break
				
				next_state, reward, gg, info = env.step(action_selected)
				g3=gg
				next_state = TO_ONE_HOT(next_state)
				next_state = torch.tensor([next_state], device=device, dtype=torch.float)
				total_reward += reward
				# print('action_selected', action_selected)
				# print(env.get_board())
				# print('RL_reward', reward)
				# print('next_state', next_state)
				if g1 and g2 and g2:
					break
				current_state = next_state
				env.check_legal_step()
				# env.render()
			print('RL_model_reward', total_reward)
			print('random_model_reward', random_model.get_total_reward())
			print('greedy_model_reward', greedy_model.get_total_reward())

			ary_rd.append(random_model.get_total_reward())
			ary_rl.append(total_reward)
			ary_gd.append(greedy_model.get_total_reward())
			pd.DataFrame(dict({'RL':ary_rl,'Random':ary_rd,'Greedy':ary_gd})).to_csv("output_best_1000_20")
		else:
			print('The checkpoint DOES NOT exist!!')
def test_2():
	f = open(f'{DIRPATH}checkpoint.txt',mode = 'r')
	checkpoint=int(f.read())-1
	ary_gd=[]
	ary_rl=[]
	ary_rd=[]
	idx = checkpoint
	score_rl=score_rd=score_gd=0
	checkpoint_path = os.path.join(DIRPATH, f'checkpoint_{idx}.pt')
	print(checkpoint_path)
	for __ in range(5):
		env = Match3Env()
		env.reset()
		env.check_legal_step()
		env_random = copy.deepcopy(env)
		random_model = random_baseline(env_random)


		env_greedy = copy.deepcopy(env)
		greedy_model = greedy_baseline(env_greedy)

		n_action = env.action_space.n
		test_network = Q_network(n_action).to(device=device)
		
		
		if os.path.exists(checkpoint_path):
			test_network.load_state_dict(torch.load(checkpoint_path))
			current_state = env.get_board()
			current_state = TO_ONE_HOT(current_state)
			current_state = torch.tensor([current_state], device=device, dtype=torch.float)
			total_reward = 0
			cnt = 1
			while True:
				g1,g2,g3=False,False,False

				random_model.execute()
				if random_model.game_over():
					g1=True
				greedy_model.execute()
				if greedy_model.game_over():
					g2=True
				# action_selected = test_network(current_state).max(1)[1].cpu().detach().numpy()[0]
				# action_selected = torch.tensor(action_selected, device=device)
				
				action_selected_N = torch.argsort(test_network.forward(current_state))
				action_selected = torch.argmax(test_network.forward(current_state))
				# print('action_selected_N', action_selected_N)
				for action in action_selected_N[0]:
					# print('action', action)
					# print(env.legal_action_index)
					if action in env.legal_action_index:
						
						action_selected=action
						break
				cnt += 1
				next_state, reward, gg, info = env.step(action_selected)
				g3=gg
				next_state = TO_ONE_HOT(next_state)
				next_state = torch.tensor([next_state], device=device, dtype=torch.float)
				total_reward += reward
				# print('action_selected', action_selected)
				# print(env.get_board())
				# print('RL_reward', reward)
				# print('next_state', next_state)
				if g1 and g2 and g2:
					break
				current_state = next_state
				env.check_legal_step()
				# env.render()
			# print('RL_model_reward', total_reward)
			# print('random_model_reward', random_model.get_total_reward())
			# print('greedy_model_reward', greedy_model.get_total_reward())
			score_rl+=total_reward
			score_rd+=random_model.get_total_reward()
			score_gd+=greedy_model.get_total_reward()
		else:
			print('The checkpoint DOES NOT exist!!')
	ary_rd.append(score_rd/5)
	ary_rl.append(score_rl/5)
	ary_gd.append(score_gd/5)
	csv=dict()
	csv['RL']=ary_rl
	csv['Random']=ary_rd
	csv['Greedy']=ary_gd
	print(score_rd/5, score_gd/5, score_rl/5)
		# pd.DataFrame(csv).to_csv("output_10.csv")


if __name__ == "__main__":
	if TEST:
		test_2()
	else:
		train()
