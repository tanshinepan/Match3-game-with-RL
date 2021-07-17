import gym
from gym import utils , error , spaces
from gym.utils import seeding
from match3 import Match3_Game
from config import *
from matrix_like import *
import numpy as np
from itertools import product
import cv2
import os

class Match3Env(gym.Env):
	def __init__(self):
		self.max_steps=MAX_STEPS
		self.output_matrix=OUTPUT_MATRIX
		self.numOfCube=NUMOFCUBE
		self.level=LEVEL
		self.diagonal_detect=DIAGONAL_DETECT
		self.exchange_only_with_neighbor=EXCHANGE_ONLY_WITH_NEIGHBOR
		self.combo_bonus=COMBO_BONUS
		self.board_length=LENGTH
		self.match_length_bonus=MATCH_LENGTH_BONUS
		self.game=Match3_Game(	LEVEL=self.level,
								data=game_matrix(),
								DIAGONAL_DETECT=self.diagonal_detect,
								NUMOFCUBE=self.numOfCube,
								MAX_STEPS=self.max_steps,
								OUTPUT_MATRIX=self.output_matrix,
								EXCHANGE_ONLY_WITH_NEIGHBOR=self.exchange_only_with_neighbor,
								COMBO_BONUS=self.combo_bonus,
								MATCH_LENGTH_BONUS=self.match_length_bonus
								)

		self.__match3_actions=self.get_available_actions()
		self.legal_actions=[]
		self.legal_action_index=[]
		self.greedy_baseline_action=0
		self.action_space=spaces.Discrete(len(self.__match3_actions))
		self.__match3_action_states=np.array([[]])
		self.cubes_img,self.board_img=self.load_image()
		self.observation_space = spaces.Box(
			low=0.0,
			high=255.0,
			#shape=self.__game.board.board_size,
			shape=(self.board_length,self.board_length,3),
			dtype=np.float64
			)
	def action_num(self):
		return len(self.__match3_actions)
	def points_generator(self):
		rows=cols=self.level
		points = [(i, j) for i, j in product(range(rows), range(cols))]
		return points
	def get_available_actions(self):
		""" calculate available actions for current board sizes """
		actions = set()
		#directions = self.get_directions(board_ndim=BOARD_NDIM)
		all_points=self.points_generator()
		for point in all_points:
			right = point[0]+ 1, point[1] + 0
			down = point[0] + 0, point[1] + 1
			if right in all_points:
				actions.add(frozenset((point,right)))
			if down in all_points:
				actions.add(frozenset((point,down)))
		# for point in self.points_generator():
		# 	for axis_dirs in directions:
		# 		for dir_ in axis_dirs:
		# 			dir_p = Point(*dir_)
		# 			new_point = point + dir_p
		# 			try:
		# 				_ = self.__game.board[new_point]
		# 				actions.add(frozenset((point, new_point)))
		# 			except OutOfBoardError:
		# 				continue
		return list(actions)
		##not finished 
	def load_image(self):
		img=[]
		path="image/"
		for i in range(self.numOfCube):
			image=cv2.imread(os.path.join(path+str(i)+'0.png'),cv2.IMREAD_UNCHANGED  )
			# print(image.shape)
			image=image[:,:,0:3]
			image=cv2.resize(image,(CUBE_LENGTH,CUBE_LENGTH),interpolation=cv2.INTER_CUBIC)
			img.append(image)
		img_board=np.full([84,84,3],0)

		# print(img_board.shape)
		
		return img,img_board	
	def reset(self):
		self.game.start()
		# self.legal_actions,self.greedy_baseline_action=self.game.get_legal_steps(self.__match3_actions)
		return self.get_current_state()
	def render(self):
		img=self.board_img
		# print(self.get_board())
		for i in range(self.level):
			for j in range(self.level):
				img[i*CUBE_LENGTH:(i+1)*CUBE_LENGTH,j*CUBE_LENGTH:(j+1)*CUBE_LENGTH]=self.cubes_img[ self.get_board()[i][j]]
		# cv2.imwrite("current_state.png",img)
		# print(img.shape)
		return img
		# cv2.imshow('img',img)
		# print(img.shape)
		# cv2.waitKey(1)

	def get_action(self, ind):
		return self.__match3_actions[ind]
	def get_state(self,ind):
		return self.__match3_action_states[ind]
	def get_board(self):
		return self.game.data.sand_board
	def check_legal_step(self):
		self.legal_actions,self.greedy_baseline_action,self.legal_action_index=self.game.get_legal_steps(self.__match3_actions)
		if(len(self.legal_actions)==0):
			return True
		return False
	def step(self,action):
		ac=self.get_action(action)
		p1,p2 = ac
		# print(p1,p2)
		reward = self.swap(p1,p2)
		if self.game.left_step() >0:
			game_over = False
		else:
			game_over = True
		if reward == 0 and TEST==False:
			game_over=True
		# ob = self.game.data.sand_board[np.newaxis,:]
		ob = self.get_current_state()
		
		return ob,reward ,game_over,{}
	def get_current_state(self):
		state=self.render()
		return state
	def return_all_states(self):
		return self.__match3_action_states
	def get_all_states(self):
		all_state=[]
		board=self.get_board().copy()
		board_tmp=board.copy()
		for i in range(len(self.__match3_actions)):
			p1,p2=self.get_action(i)
			p11,p12=p1
			p21,p22=p2
			# board[p11][p12],board[p21][p22]=board[p21][p22],board[p11][p12]
			board_tmp[p1],board_tmp[p2]=board[p2],board[p1]
			tmp=list([])
			tmp.append(board)
			tmp.append(board_tmp)
			all_state.append(np.array(tmp))
			board_tmp=board.copy()
		self.__match3_action_states=np.array(all_state)
		return self
	def swap(self, point1 , point2):
		try :
			reward = self.game.move(point1,point2)
		except:
			reward = 0
		return reward

if __name__=="__main__":
	a=Match3Env()
	state=a.reset()
	print(a.get_current_state().shape)
	# print(state)

