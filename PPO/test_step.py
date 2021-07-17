from match3Env import Match3Env
import numpy as np
import pandas as pd
import os
import copy
import shutil
import json
import sys
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.rllib.utils.numpy import softmax
from non_model import random_baseline, greedy_baseline
from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

checkpoint_root = 'checkpoint_5'
ray_results = f'{checkpoint_root}/ray_results/'
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
info = ray.init(ignore_reinit_error=True, num_gpus=1)

trainer_config = DEFAULT_CONFIG.copy()

# trainer_config['num_workers'] = 4
trainer_config["train_batch_size"] = 400
trainer_config["sgd_minibatch_size"] = 64
trainer_config["num_sgd_iter"] = 10

def env_creator(env_config):
    return Match3Env()  # return an env instance

register_env('Match3Env', env_creator)

Match3EnvClass = 'Match3Env'
config = trainer_config
trainer = PPOTrainer(config, Match3EnvClass)
if os.path.exists(checkpoint_root):
    checkpoints = os.listdir(checkpoint_root)
    for i in range(len(checkpoints)):
        checkpoints[i] = int(checkpoints[i].replace('checkpoint_', ''))
    checkpoint = max(checkpoints)


rnd_score_list = []
gdy_score_list = []
ppo_score_list = []
interval = 100
legal_action = True

checkpoint_idx = checkpoint - 1
trainer.restore(os.path.join(checkpoint_root, 'checkpoint_%06d' % checkpoint_idx, f'checkpoint-{checkpoint_idx}'))
time = 5
rnd_score = 0
gdy_score = 0
ppo_score = 0
for _ in range(time):
    env = Match3Env()
    state = env.reset()
    if legal_action:
        env.check_legal_step()
    cond = TEST == True and SWAP_AGAIN == False
    cond = True

    env_random = copy.deepcopy(env)
    random_model = random_baseline(env_random)
    if cond:
        while True:
            random_model.execute()
            if random_model.game_over():
                break
    else:
        random_model.execute()
    rnd_score += random_model.get_total_reward()

    env_greedy = copy.deepcopy(env)
    greedy_model = greedy_baseline(env_greedy)
    if cond:
        while True:
            greedy_model.execute()
            if greedy_model.game_over():
                break
    else:
        greedy_model.execute()
    gdy_score += greedy_model.get_total_reward()

    done = False
    max_state = -1
    cumulative_reward = 0
    reward = 1
    # while not done:
    cnt = 0
    while not done:
        if legal_action:
            env.check_legal_step()
        action = trainer.compute_action(state)
        res = trainer.compute_action(state, full_fetch=True)
        actions = res[2]["action_dist_inputs"]
        actions = np.flip(np.argsort(actions))
        for action_id in actions:
            if action_id in env.legal_action_index:
                action = action_id
                break
        state, reward, done, results = env.step(action)
        cumulative_reward += reward
    ppo_score += cumulative_reward
rnd_score /= time
gdy_score /= time
ppo_score /= time
rnd_score_list.append(rnd_score)
gdy_score_list.append(gdy_score)
ppo_score_list.append(ppo_score)
csv = dict()
csv['Random'] = rnd_score_list
csv['Greedy'] = gdy_score_list
csv['PPO'] = ppo_score_list
print('rnd_score', rnd_score)
print('gdy_score', gdy_score)
print('ppo_score', ppo_score)
# pd.DataFrame(csv).to_csv(f"new_output_{interval}_test_{TEST}_swap_{SWAP_AGAIN}_legal_{legal_action}.csv")