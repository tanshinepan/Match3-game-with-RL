# Match3-game-with-RL
## 1. Abstact
With the development of the artificial intelligence area, an increasing number of algorithms in the deep reinforcement learning area  are created. Moreover, there are new challenges for their comprehensive analysis and searching application areas. We wonder if the deep reinforcement learning model will perfectly consider the technique and some luck component. The Match-3 game, which has simple gameplay, but some random event the players can’t control. Thus, we choose to test a deep reinforcement learning model on the match-3 games to see how many controlled and uncontrolled factors and patterns it can learn. Although the game of go is more complicated, match-3 game is more interesting to be applied to data driven learning models, since its score is accompanied with the probability. If it can learn very well, it means it can precisely predict some random event. The article provides metrics for evaluation of agents and corresponding baselines in different scenarios. If the model can solve not only computational problems but also tasks with chance, We are glad to see how it can help people in the future.
## 2.	Introduction
With the advance of deep reinforcement learning, we expect more and more regions can be applied to make more precise decisions. Alphago, the worldwide famous model, beats humans on the game of go and proves that its computation ability is better than people’s. So far, we increasingly depend on the data and the model when we face the tasks or make a choice. Unfortunately, compared with our millions of problems, we lack lots of data and the opportunity to be supported with the machine. The scope of their application is limited. Although the technique is advanced, we don’t know what industry can combine with deep reinforcement learning, and the lack of data could cause the prediction isn’t precise enough. Thus, we decided to apply it to the games.

Match-3 games became very popular last year, they are top grossing in AppStore and Google Play. That’s because it is a simple and interesting game and we can easily play it by manipulating it on the smartphone. However, although it is easy to get high grades, if you want to gain more scores, you need the assistance of luck. Obviously, it is very different from the game of go, since the latter only considers the computation ability but the former is more depending on the chance. In the real world, we almost never assert something will happen 100 percent because of its randomness. Therefore, studying on the match-3 game may be more beneficial than the game of go. In addition, for the experiment, we create lots of changeable configurations.

The previously proposed algorithms include methods such as MCTS [1], deep neural network (DNN) [2], CNN[3] don’t do very well on the decision-making game, Thus, we apply the deep reinforcement learning algorithm like DQN [4] to our game with diverse environments. Although there is some research about DRL and match-3 type games [5], [6], we still study it since our setting is very different from theirs, we adopt some interfaces, their result isn’t positive enough, and the fortune takes lots of account on our work. Moreover, we adapt the code [7] to be more elastic and suitable for our work and rewrite it to the GYM interface [8], [9], [10], which is easy to utilize for the deep reinforcement learning algorithm and experiments. Furthermore, we create 2 baselines methods to make the comparison.
### 2.1. GYM [8], [9], [10]
A toolkit for developing and comparing reinforcement learning algorithms. GYM makes no assumptions about the structure of your agent. It is compatible with any numerical computation library, such as Pytorch. We package our environment to GYM for reinforcement learning experiments and easier access.
The following is the critical method:

**env = gym.make()** - create the game environments

**observation = env.reset()** - reset the environment to default

**env.render()** - render the state and environment

**action = env.action_space.sample()** - get the action

**observation, reward, done, info = env.step(action)** - apply the action to the state, and we will get the new state, reward, if game ends, and the other information 
### 2.2. RLlib [11], [12]
A module on Python for providing reinforcement learning algorithms. We use some of its methods to implement the algorithm. The authors argue for building composable RL components by encapsulating parallelism and resource requirements within individual components, which can be achieved by building on top of a flexible task-based programming model
## 3.	Environment - Match-3 game environment 
Although our code has lots of changeable parts, we often use the default setting when we conduct the experiment.
### 3.1. Rule
* Any 2 shapes can be switched, steps would minus 1 even if no matches are found after switching.
* Matches if 3 or more of same shape are in a row, column or diagonal. 
* For n numbers of the same shape matched and connected, score of shape is added by n-2.
* Total score is the score of all shapes added together.
### 3.2. Initialization
* Creates a 6x6 plane (can be changed).
* Fill in with the 5 random shapes (can add more if necessary).
* Check to make sure no matches exist.
* If matches exist, try another shape.
* Set the numbers of the steps left and score to 100 and 0  (the number of max steps can be changed).
* Display score of each shape, total score and steps left at the bottom.
## 4.	Methodology
To create the game environment, we first write a match 3 game so it can be used in this project, in which it has to be able to accept inputs in both UI interaction via mouse clicks and in commands. For the experiment, we wish the game is elastic to change some configurations and easy to use. Thus, we adapt the game code [5] and nearly rewrite it to the brand new one. The game we create has a diversity of settings, not only the grade calculation but also the game rules can easily be changed.

After the game environment is set, we rewrite the code to the GYM interface [8], [9], [10] to be more convenient to train with deep reinforcement learning, and start to train the model with deep reinforcement learning algorithm like DDQN, PPO, and A3C, and the model of deep learning part is CNN. Moreover,we provide two baseline to be compared with and we hope the score played by the learning based model can beat that of the baseline and that we play.
### 4.1.	Process
![](https://user-images.githubusercontent.com/43957213/127611031-336496c9-436f-47b1-8424-c6ca72209554.png)
### 4.2. RL framework
* Algorithm: DDQN, PPO, A3C
* Action: The core mechanic of the game allows to swap two arbitrary shapes on the board
* Environment: All the matrices and the game board.
* Reward: For the current state, the action just is taken and the next state of the board reward is the number of deleted shapes minus 2. Each action of the agent should lead to reward.
### 4.3.	Changes made on the game environment
Some changes were made to package the environment for reinforcement learning experiments.
* Plane was set to 8x8
* Number of shapes was changed from 5 to 4
* Rule was changed from swappable between any position to only swappable with arbitrary shapes
### 4.4.	RL implementation
#### 4.4.1.	Self implemented
* DQN
* DDQN

We want to ensure the algorithm is correct and the code is suitable for our experiment. Furthermore, we hope the code is elastic to use. So we build the code with Pytorch instead of using other higher level modules. DQN and DDQN are trained with legal actions since normal actions are very unfriendly to value-based models.
#### 4.4.2.	Using RLlib [11], [12]
* PPO
* A3C

Since we think the policy-based model is more complicated to implement and harder to debug than value-based, we import the RLlib  [11], [12] module on Python for correctness. The RLlib [11], [12] module provides lots of deep reinforcement learning algorithms and friendly connections to the GYM interface [8], [9], [10]. PPO and A3C are trained with normal actions since we regard that the difference between normal actions and legal actions isn’t larger than value-based models.
#### 4.4.3.	DQN [4] and DDQN Model
* 2 Convolution layer – checking the line and other patterns
* Various activate functions – model diversity
* Dropout layer – not focusing on training particular parts.
* Fully connective layer – considering more parameters.
* One Hot – the scores depend on patterns instead of the shape of the cubes.
#### 4.4.4.	Action
* Normal actions – use argmax only to get swaps between different shapes. The model is hard to train with normal actions because of its diversity and convergence difficulty.  
* Legal actions – use argsort and checking to get swaps between different shapes with matching results. The performance is better because of its restriction, and it is easy to converge.
## 5.	Experiment 
### 5.1. Baseline
We set 2 baselines to compare with our model to see if the model is good enough and useful. The random baseline means the agent takes random action, and the greedy baseline represents that the agent makes the best choice without considering the lusk part. Our goal is to train a model that performs better than the random model and as close to the greedy model as possible
### 5.2. Results
#### Scores vs time
The performances of three deep reinforcement learning algorithms as the time passes.

![](https://user-images.githubusercontent.com/43957213/126825787-74f71017-e7e4-4fd6-a8f5-668511e80382.png)
![](https://user-images.githubusercontent.com/43957213/126825790-74410964-a833-49ce-9c20-4263e4f58ae3.png)
![](https://user-images.githubusercontent.com/43957213/126825792-0573f675-2b71-4353-a6ee-5a0d3c86e93b.png)
#### Relative score vs time (compare to random baseline)
The relative performances of three deep reinforcement learning algorithms beyond the random baseline as the time passes.

![](https://user-images.githubusercontent.com/43957213/126825795-114b90b9-1143-4456-85ab-328fb9b1193b.png)
![](https://user-images.githubusercontent.com/43957213/126825798-84367946-3049-459d-bec7-5887aa2d6bdc.png)
#### Score vs steps
The performances of three deep reinforcement learning algorithms as the numbers of the training step increase.

![](https://user-images.githubusercontent.com/43957213/126824340-475fd4ef-28df-47be-9e1a-770780260535.png)
![](https://user-images.githubusercontent.com/43957213/126825801-db58f25b-41ab-4591-89bf-d36bee86bc02.png)
![](https://user-images.githubusercontent.com/43957213/126826117-4fb29439-201f-4e36-8b1f-bcb3b217d08c.png)
#### Relative score vs steps (compare to random baseline)
The relative performances of three deep reinforcement learning algorithms beyond the random baseline as the numbers of the training step increase.

![](https://user-images.githubusercontent.com/43957213/126825805-8ea00927-9fbb-404c-a296-073716623d6e.png)
![](https://user-images.githubusercontent.com/43957213/126824341-d64c1835-f4c4-46d2-80ac-3ebc48b7a965.png)

The performance of three deep reinforcement learning algorithms isn’t good enough. All of them don’t beat the baselines, the model can’t help the player to win.
## 6.	Conclusion
All of the deep reinforcement learning models don’t reach our expectation, and another paper [5] shows that DQN and PPO can’t beat the random baseline. Thus, we conclude that DRL may benefit match-3 games a lot. Moreover, because the game has too many states (# states = # shapes ^ (row * column)), the game may be too complicated to be trained well. And we may haven’t trained enough steps and time to converge  yet before we test cause the bad performance. Furthermore, training with legal actions is time efficient but more complex due to dynamic action space and requires masking, and we only put legal masks on the models we built by ourselves. For the models with legal actions, it is very complicated in dynamic action space, and for the models without legal action masks, it may take lots of time to converge and be useful. Unfortunately, both reasons lead to the worse performance. Above all, The most critical reason is our game environment setting, our games with 8 * 8 planes and 5 shapes take too large a part of the luck. It is approximately a fortune game instead of a technique game. Thus, we have to correct our wrong game configurations and take other tests in the future.
## 7.	Future Use
Seeing that the performance of three models isn’t good, we have to use the more powerful model in the deep learning part. Therefore, we should try some strong models like RestNet. Moreover, because we have to complete the easier task before finishing the harder problem, we should restrict legal actions on every model instead of utilizing normal actions. In addition, we have to adopt other RL algorithms and find which is the most appropriate to suit the match-3 games. Last but not least, the bad performance of the models may be caused by the large impact from the luck, so we should adjust the configurations and parameters to ensure the game won’t take great account of the fortune.
## 8.	Reference
[1] POROMAA, Erik Ragnar. Crushing candy crush: predicting human success rate in a mobile game using Monte-Carlo tree search. 2017. 

[2] PURMONEN, Sami. Predicting game level difficulty using deep neural networks. 2017. 

[3] GUDMUNDSSON, Stefan Freyr, et al. Human-like playtesting with deep learning. In: 2018 IEEE Conference on Computational Intelligence and Games (CIG). IEEE, 2018. p. 1-8. 

[4] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html?fbclid=IwAR1UzQffW67bR-NAFrVGjeNNH93jeCT77nPyrdxM5aUz1Vx5TUkW_1gfKWE

[5] KAMALDINOV, Ildar; MAKAROV, Ilya. Deep reinforcement learning in match-3 game. In: 2019 IEEE conference on games (CoG). IEEE, 2019. p. 1-4.

[6] SHIN, Yuchul, et al. Playtesting in match 3 game using strategic plays via reinforcement learning. IEEE Access, 2020, 8: 51593-51600.

[7] https://github.com/wrk226/match_3_game_simulator_py

[8] BROCKMAN, Greg, et al. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

[9] https://gym.openai.com/docs/?fbclid=IwAR28D5AQlDWBo_UJ-3U2oFcCW_yz_t9wx1of0TQzKJK6PJfTfI0W6hkpRxI

[10] https://github.com/hans1996/match3_env?fbclid=IwAR315g0k8qYmJAjt12FuZ6ycLwwRiK3b1Gt63l9EZ0nVdefOupo2tiy8DeU

[11]IANG, Eric, et al. RLlib: Abstractions for distributed reinforcement learning. In: International Conference on Machine Learning. PMLR, 2018. p. 3053-3062.

[12] htps://docs.ray.io/en/master/rllib.html?fbclid=IwAR2X94QIGzZ18ryzaL4dg908maxORCQtNjqtfTY83t-s_nWW-F_4SWjTORA
