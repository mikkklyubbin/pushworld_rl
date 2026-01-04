import sys
import os
import pushworld
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/home/mik/hse/Pushworld/pushworld-main/python3/src/pushworld'))
from stable_baselines3.common.evaluation import evaluate_policy
from pushworld.gym_env import PushTargetEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import matplotlib.pyplot as plt
import numpy as np
import dataframe_image as dfi
import cv2
import pandas as pd
from pushworld.gym_env import savergb
from pushworld.model import train_ppo
model_save_path = "/home/mik/hse/Pushworld/pushworld-main/python3/model/bst2/best_model.zip"


id:int = 0
cool_table = pd.DataFrame({'Type':[], 'Test%':[], 'Train%':[]})
#, "walls", "shapes", "base", "obstacles", "goals"
for group in ["all", "walls", "shapes", "base", "obstacles", "goals"]:
    print(group)
    test_env = PushTargetEnv(f"/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/{group}/test", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True)
    train_env = PushTargetEnv(f"/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/{group}/train", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True)
    model = train_ppo(train_env, None, 100)

    num_episodes = 200
    success_count = 0
    for episode in range(num_episodes):

        obs, _ = test_env.reset()  
        terminated = False
        truncated = False
        episode_rewards = []
        while not terminated:
            action, _ = model.predict(obs)  

            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_rewards.append(reward)
            if (truncated):
                break
            
        if terminated:
            rgb = test_env.render()
            # savergb(rgb, "/home/mik/hse/Pushworld/pushworld-main/python3/fotos/" + str(episode) + ".jpg")
            success_count += 1
    print(f"\nРезультаты за {num_episodes} эпизодов:")
    print(f"Успешных эпизодов: {success_count}")
    print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
    s1 = success_count/num_episodes*100

    test_env = PushTargetEnv(f"/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/{group}/train", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True)

    num_episodes = 200
    success_count = 0
    for episode in range(num_episodes):

        obs, _ = test_env.reset()  
        terminated = False
        truncated = False
        episode_rewards = []
        while not terminated:
            action, _ = model.predict(obs)  

            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_rewards.append(reward)
            if (truncated):
                break
            
        if terminated:
            rgb = test_env.render()
            savergb(rgb, "/home/mik/hse/Pushworld/pushworld-main/python3/fotos/" + str(episode) + ".jpg")
            success_count += 1
    print(f"\nРезультаты за {num_episodes} эпизодов:")
    print(f"Успешных эпизодов: {success_count}")
    print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
    s2 = success_count/num_episodes*100
    cool_table.loc[id] = [group, s1, s2]
    id += 1

dfi.export(cool_table, '/home/mik/hse/Pushworld/pushworld-main/benchmark/tables/eval_PushTarget.png')