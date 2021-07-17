import numpy as np
import os
import random 
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from match3Env import Match3Env
from config import *
import time
# from Q_learning import Buffer
class random_baseline:
	def __init__(self, env=Match3Env().reset()):
		self.__env = env
		# self.table = Buffer()
		self.__current_state=self.__env.get_board()
		self.__reward=.0
		self.__total_reward=.0
		self.__gg=False
		self.__info={}
		self.__next_state=[]
	
	def execute(self):
		self.__next_state,self.__reward,self.__gg,self.__info=self.__env.step(self.random_select())
		self.__total_reward+=self.__reward
		self.__current_state=self.__next_state
		# self.__env.render()
	def game_over(self):
		return self.__gg
	def get_current_state(self):
		return self.__current_state
	def get_info(self):
		return self.__info
	def get_current_reward(self):
		return self.__reward
	def get_total_reward(self):
		return self.__total_reward
	def random_select(self):
		self.__env.check_legal_step()
		while True:
			action_selected = random.randint(0, self.__env.action_num() - 1)

			if action_selected in self.__env.legal_action_index:
				return action_selected
		
class greedy_baseline:
	def __init__(self, env=Match3Env().reset()):
		self.__env = env
		# self.table = Buffer()
		self.__current_state=self.__env.get_board()
		self.__reward=.0
		self.__total_reward=.0
		self.__gg=False
		self.__info={}
		self.__next_state=[]
	
	def execute(self):
		self.__next_state,self.__reward,self.__gg,self.__info=self.__env.step(self.select_action())
		self.__total_reward+=self.__reward
		self.__current_state=self.__next_state
		# self.__env.render()
	def game_over(self):
		return self.__gg
	def get_current_state(self):
		return self.__current_state
	def get_info(self):
		return self.__info
	def get_current_reward(self):
		return self.__reward
	def get_total_reward(self):
		return self.__total_reward
	def select_action(self):
		self.__env.check_legal_step()
		return self.__env.greedy_baseline_action
def main():
	env=random_baseline()
	while True:
		env.execute()
if __name__=="__main__":
	main()