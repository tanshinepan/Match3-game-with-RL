import sys
# import pygame
from matrix_like import *
import copy
# from pygame.transform import scale
# from  config import *

class Match3_Game:
	def __init__(self,
				LEVEL=6,
				data=game_matrix(),
				OUTPUT_MATRIX=True,
				NUMOFCUBE=8,
				DIAGONAL_DETECT=False,
				EXCHANGE_ONLY_WITH_NEIGHBOR=True,
				LENGTH=500,
				WHITE= (255, 255, 255),
				RED= (255, 0, 0),
				BLACK= (0,0,0),
				MAX_STEPS=100,
				DROP_SPEED = 100,
				FLASH_SPEED = 10,
				FLASH_TIMES = 0,
				WIN_WIDTH = 600,
				WIN_HEIGHT = 650,
				BORDER_WIDTH = 5,
				COMBO_BONUS=1,
				MATCH_LENGTH_BONUS=.01):
		##config
		self.LEVEL=LEVEL
		self.OUTPUT_MATRIX=OUTPUT_MATRIX
		self.DIAGONAL_DETECT=DIAGONAL_DETECT
		self.NUMOFCUBE=NUMOFCUBE
		self.BORDER_WIDTH=BORDER_WIDTH
		self.EXCHANGE_ONLY_WITH_NEIGHBOR=EXCHANGE_ONLY_WITH_NEIGHBOR
		self.LENGTH=LENGTH
		self.DELTA= round(self.LENGTH / self.LEVEL)
		self.CUBE_LENGTH=int(self.LENGTH / self.LEVEL - self.BORDER_WIDTH)
		self.WHITE=WHITE
		self.RED=RED
		self.BLACK=BLACK
		self.MAX_STEPS=MAX_STEPS
		self.DROP_SPEED=DROP_SPEED
		self.FLASH_SPEED=FLASH_SPEED
		self.FLASH_TIMES=FLASH_TIMES
		self.WIN_WIDTH=WIN_WIDTH
		self.WIN_HEIGHT=WIN_HEIGHT
		##Env info
		self.data=data
		self.score=0
		self.reward=0.0
		self.current_step=self.MAX_STEPS
		self.COMBO_BONUS=COMBO_BONUS
		self.MATCH_LENGTH_BONUS=MATCH_LENGTH_BONUS
		
		##TODO: initialize the game board, score ...
	def start(self):
		self.score=0
		self.data=game_matrix()
		self.current_step=self.MAX_STEPS
		return self

	##TODO: Swap points 
	def move(self,point1:tuple,point2:tuple):
		self.reward=0
		if self.EXCHANGE_ONLY_WITH_NEIGHBOR==True:
			det=abs(point1[0]-point2[0])+abs(point1[1]-point2[1])
			if det != 1:
				self.current_step-=0
				return self.reward
			else:
				self.current_step-=1
				self.data.sand_board[point1],self.data.sand_board[point2]=self.data.sand_board[point2],self.data.sand_board[point1]
				return self.match(point1,point2)

		else:
			self.current_step-=1
			self.data.sand_board[point1],self.data.sand_board[point2]=self.data.sand_board[point2],self.data.sand_board[point1]
			
			return self.match(point1,point2)
	##TODO: search the matches
	def match(self,point1,point2):
		combo=0
		temp_cube_matrix, pair_lists = self.data.match_once(((point1), (point2)), 1)
		##if no match to exchange again
		if(SWAP_AGAIN):
			if(len(pair_lists)==0):
				self.data.sand_board[point1],self.data.sand_board[point2]=self.data.sand_board[point2],self.data.sand_board[point1]
				return 0.0
			combo+=len(pair_lists)
		for i in range(len(pair_lists)):
			self.reward+=pow(self.MATCH_LENGTH_BONUS,len(pair_lists[i])-3)
		while np.count_nonzero(self.data.sand_board == self.NUMOFCUBE) != 0:
					drop_record = self.data.move_down(self.data.sand_board)
					while True:
						self.data.get_new_cubes(self.data.sand_board, False)
						drop_record = self.data.move_down(self.data.sand_board)
						if np.count_nonzero(self.data.sand_board == self.NUMOFCUBE) == 0:
							break
					temp_cube_matrix, pair_lists = self.data.match_once((), 2)
					combo+=len(pair_lists)
					for i in range(len(pair_lists)):
						self.reward+=pow(self.MATCH_LENGTH_BONUS,len(pair_lists[i])-3)
				# 	print(self.data.sand_board)
				# print(self.data.cube_matrix)
		self.reward*=1+(combo*self.COMBO_BONUS)
		self.score +=self.reward
		return self.reward
	def left_step(self):
		return self.current_step
	def match_nonswap(self,point1,point2):
		self.data.sand_board[point1],self.data.sand_board[point2]=self.data.sand_board[point2],self.data.sand_board[point1]
		temp_cube_matrix, pair_lists = self.data.match_once(((point1), (point2)), 1)
		if(len(pair_lists)==0):
			return False,0
		combo=0
		combo+=len(pair_lists)
		count=1
		for i in range(len(pair_lists)):
			count*=pow(self.MATCH_LENGTH_BONUS,len(pair_lists[i])-3)
		count=count*combo
		return True , count
	def get_legal_steps(self,all_action_array):
		temp_action_array=set()
		temp_action_index=[]
		idx,max_goal=0,0.0
		for i in range(len(all_action_array)):
			board_copied=copy.deepcopy(self)
			p1,p2=all_action_array[i]
			matched,goal=board_copied.match_nonswap(p1,p2)
			if(matched):
				temp_action_array.add(frozenset((p1,p2)))
				temp_action_index.append(i)
				if(goal>=max_goal):
					idx=i
					max_goal=goal
		return list(temp_action_array),idx,list(temp_action_index)
def main():
	a=Match3_Game()
	a.start()
	# a.start_game()
	print(a.data.sand_board)
	a.move(point1=(0,0),point2=(0,1))

	print(a.data.sand_board)

if __name__ == '__main__':
	main()
