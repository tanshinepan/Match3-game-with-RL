import numpy as np

# 沙盤參數
WIN_WIDTH = 600
WIN_HEIGHT = 650
BORDER_WIDTH = 5

# animate config
DROP_SPEED = 100
FLASH_SPEED = 10
FLASH_TIMES = 0

# game config

MAX_STEPS = 5000
OUTPUT_MATRIX = True  		##True: print cube_matrix False: draw cube_matrix
NUMOFCUBE = 4				##number of cube type. If #>9 then output matrix is False
LEVEL = 8					##board = LEVEL * LEVEL
DIAGONAL_DETECT = False		##check diagonal line 
EXCHANGE_ONLY_WITH_NEIGHBOR = True
COMBO_BONUS = 0.1
MATCH_LENGTH_BONUS = 1.01
# board's length/width
LENGTH = 500
DELTA = round(LENGTH / LEVEL)
CUBE_LENGTH = int(LENGTH / LEVEL - BORDER_WIDTH)

# color config
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0,0,0)


# RL config
BATCH_SIZE = 128
BUFFER_CAPACITY = 50000
BUFFER_UPDATE_FRACTION = .25
BUFFER_ACCESS_FRACTION = 1
SELECT_ACTION_METHOD = 1
EPSILON_GREEDY_INIT = .15
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
GAMMA = 0.999
N_EPISODES = 10000000
# STEP_UPDATE_NETWORK = 10000
STEP_UPDATE_NETWORK = 100       ##training one time/ 128 steps . update frequency = 128*100 steps
CHECK_LEGAL_STEPS = 1000
EPISODE_DISPLAY = 1
EPISODE_UPDATE_NETWORK = 10
EPISODE_SAVE_MODEL = 1
SWAP_AGAIN = True
RENDER = False
DIRPATH = "./model_5_gg/"
PRINT_STEP=False
TEST=False   ## if not test reward == 0 ? gg: continue
if TEST==True:
    MAX_STEPS = 120
