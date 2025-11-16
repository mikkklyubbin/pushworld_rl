import sys
import os
import pushworld
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from pushworld.gym_env import PushWorldEnv
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/home/mik/hse/Pushworld/pushworld-main/python3/src/pushworld'))




model = PPO.load("/home/mik/hse/Pushworld/pushworld-main/python3/model/ppo_graph.zip")

test_env = PushWorldEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/all/test", 100, to_height = 11, to_width = 11, seq = True, need_pddl = True)
print(test_env.reset())
import time

num_episodes = 10
success_count = 0
for episode in range(num_episodes):
    
    obs, _ = test_env.reset()  
    terminated = False
    truncated = False
    episode_rewards = []
    while not terminated:

        st = time.time()
        test_env.get_relations_graph()
        et = time.time()
        action, _ = model.predict(obs)

        # считаем время исполнения
        elapsed_time = et - st
        print('Время исполнения:', elapsed_time, 'секунд')
        obs, reward, terminated, truncated, info = test_env.step(action)
        if (terminated):
            print(11)
        episode_rewards.append(reward)
        if (truncated):
            break
    if terminated:
        print(1)
        rgb = test_env.render()
        
        success_count += 1
print(f"\nРезультаты за {num_episodes} эпизодов:")
print(f"Успешных эпизодов: {success_count}")
print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")