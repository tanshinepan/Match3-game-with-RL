# Match3-game-with-RL
## Motivation
* Write a match 3 game so it can be used in this project, in which it has to be able to accept inputs in both UI interaction via mouse clicks and in commands.
* To allow computer to play a game a match 3 by training it using reinforcement learning, and the score will beat that we play.
## Process
![](https://user-images.githubusercontent.com/43957213/126824221-f968842c-3a58-48aa-9b71-717a94e68fdf.png)
## Environment 
### Match-3 game environment
#### Rule
* Any 2 shapes can be switched, steps would minus 1 even if no matches are found after switching.
* Matches if 3 or more of same shape are in a row, column or diaganol. 
* For n number of same shape matched and connected, score of shape is added by n-2.
* Total score is the score of all shapes added together.
#### Initialization
* Creates a 6x6 plane (can be changed).
* Fill in with the 5 random shapes (can add more if necessary).
* Check to make sure no matches exist.
* If matches exist, try another shape.
* Set num. of steps left and score to 100 and 0 (num. of max step can be changed).
* Display score of each shape, total score and steps left at the bottom.
### GYM
* A toolkit for developing and comparing reinforcement learning algorithms
* Makes no assumptions about the structure of your agent
* Compatible with any numerical computation library, such as Pytorch
* Package our environment to GYM for reinforcement learning experiment and easier access
* env = gym.make()
* observation = env.reset()
* env.render()
* action = env.action_space.sample() 
* observation, reward, done, info = env.step(action)
## Approach
### RL framework
* Algorithm: DDQN, PPO, A3C
* Action: The core mechanic of the game allows to swap two arbitrary shapes on the board
* Environment: All the matrices and the game board.
* Reward: For the current state, the action just is taken and the next state of the board reward is the number of deleted shapes minus 2. Each action of the agent should lead to reward.
### Changes made
Some changes were made to package the environment for reinforcement learning experiment.
* Plane was set to 8x8
* Number of shapes was changed from 5 to 4
* Rule was changed from swappable between any position to only swappable with arbitary shapes
### RL implementation
#### Self implemented
* DQN
* DDQN
#### Using Rllib
* PPO
* A3C
#### DDQN Model
* 2 Convolution layer
* Various activate functions
* Dropout layer
* Fully connective layer
* One Hot
#### Action
* Normal actions – use argmax only to get swaps between different shapes.
* Legal actions – use argsort and checking to get swaps between different shapes with matching results.
## Experiment 
### Baseline
* Our goal is to train a model that performs better than the random model and as close to the greedy model as possible
* Random <= RL <= Greedy
## Results
### Scores vs time
![](https://user-images.githubusercontent.com/43957213/126825787-74f71017-e7e4-4fd6-a8f5-668511e80382.png)
![](https://user-images.githubusercontent.com/43957213/126825790-74410964-a833-49ce-9c20-4263e4f58ae3.png)
![](https://user-images.githubusercontent.com/43957213/126825792-0573f675-2b71-4353-a6ee-5a0d3c86e93b.png)
### Relative score vs time (compare to random baseline)
![](https://user-images.githubusercontent.com/43957213/126825795-114b90b9-1143-4456-85ab-328fb9b1193b.png)
![](https://user-images.githubusercontent.com/43957213/126825798-84367946-3049-459d-bec7-5887aa2d6bdc.png)
### Score vs steps
![](https://user-images.githubusercontent.com/43957213/126824340-475fd4ef-28df-47be-9e1a-770780260535.png)
![](https://user-images.githubusercontent.com/43957213/126825801-db58f25b-41ab-4591-89bf-d36bee86bc02.png)
![](https://user-images.githubusercontent.com/43957213/126826117-4fb29439-201f-4e36-8b1f-bcb3b217d08c.png)
### Relative score vs steps (compare to random baseline)
![](https://user-images.githubusercontent.com/43957213/126825805-8ea00927-9fbb-404c-a296-073716623d6e.png)
![](https://user-images.githubusercontent.com/43957213/126824341-d64c1835-f4c4-46d2-80ac-3ebc48b7a965.png)
## Conclusion
* Training with legal actions is time efficient but more complex due to dynamic action space and requires masking
* PPO and A3C isn’t training enough.
* The game is too complicated to be trained well, since it has too many states. (# states = # shapes ^ (row * column))
* PPO isn’t as good as expected.
## Future Use
* Model: RestNet
* Legal actions mask
* Other RL algorithms
## Reference
* https://gym.openai.com/docs/?fbclid=IwAR28D5AQlDWBo_UJ-3U2oFcCW_yz_t9wx1of0TQzKJK6PJfTfI0W6hkpRxI
* https://github.com/hans1996/match3_env?fbclid=IwAR315g0k8qYmJAjt12FuZ6ycLwwRiK3b1Gt63l9EZ0nVdefOupo2tiy8DeU
* https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html?fbclid=IwAR1UzQffW67bR-NAFrVGjeNNH93jeCT77nPyrdxM5aUz1Vx5TUkW_1gfKWE
* https://ieeexplore.ieee.org/document/8848003/authors?fbclid=IwAR1UzQffW67bR-NAFrVGjeNNH93jeCT77nPyrdxM5aUz1Vx5TUkW_1gfKWE#authors
* https://docs.ray.io/en/master/rllib.html?fbclid=IwAR2X94QIGzZ18ryzaL4dg908maxORCQtNjqtfTY83t-s_nWW-F_4SWjTORA
* https://github.com/wrk226/match_3_game_simulator_py
