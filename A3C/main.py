from match3Env import Match3Env
import numpy as np
import os
import shutil
import json
import sys
import gym
import ray
from ray.rllib.agents.a3c import A3CTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

checkpoint_root = 'checkpoint_4'
# Where checkpoints are written:
# shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)
# Where some data will be written and used by Tensorboard below:
ray_results = f'{checkpoint_root}/ray_results/'
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
info = ray.init(ignore_reinit_error=True, num_gpus=1)

trainer_config = DEFAULT_CONFIG.copy()
# trainer_config['num_workers'] = 4
trainer_config["train_batch_size"] = 400


def do_training(Match3EnvClass, config = trainer_config, iterations=20):
    trainer = A3CTrainer(config, Match3EnvClass)
    if os.path.exists(checkpoint_root):
        checkpoints = os.listdir(checkpoint_root)
        for i in range(len(checkpoints)):
            checkpoints[i] = int(checkpoints[i].replace('checkpoint_', ''))
        checkpoint = max(checkpoints)
        trainer.restore(os.path.join(checkpoint_root, 'checkpoint_%06d' % checkpoint, f'checkpoint-{checkpoint}'))
    results = []
    episode_data = []
    episode_json = []
    for i in range(iterations):
        result = trainer.train()
        results.append(trainer)
    
        episode = {'n': i, 
                'episode_reward_min': result['episode_reward_min'], 
                'episode_reward_mean': result['episode_reward_mean'], 
                'episode_reward_max': result['episode_reward_max'],  
                'episode_len_mean': result['episode_len_mean']}
        
        episode_data.append(episode)
        episode_json.append(json.dumps(episode))
        file_name = trainer.save(checkpoint_root)

    return trainer


# trainer = do_training("CartPole-v1", config=trainer_config, iterations=20)

# env = gym.make('CartPole-v0')
# state = env.reset()

def env_creator(env_config):
    return Match3Env()  # return an env instance

register_env('Match3Env', env_creator)

trainer = do_training('Match3Env', config=trainer_config, iterations=1)

env = Match3Env()
state = env.reset()

done = False
max_state = -1
cumulative_reward = 0
reward = 1
# while not done:
while reward > 0 or not done:
    action = trainer.compute_action(state)
    state, reward, done, results = env.step(action)
    cumulative_reward += reward

print(f'Cumulative reward you received is: {cumulative_reward}. Congratulations!')