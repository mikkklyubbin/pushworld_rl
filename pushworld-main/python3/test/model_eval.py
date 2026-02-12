import sys
import os
import wandb
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
use_concentrtion:bool = False
new_actions_rew:float = 0.01
path_to_rep = "/home/mik/hse/Pushworld/pushworld-main/"
test_ac = []
train_ac = []
cur_grp = ""
def test_model(model):
    test_env = PushTargetEnv(path_to_rep + f"benchmark/puzzles/level0/all/test", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True, use_concentrtion = use_concentrtion, new_actions_rew = new_actions_rew, loop_penalty = 0.05)
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
            success_count += 1
    print(f"\nРезультаты за {num_episodes} эпизодов:")
    print(f"Успешных эпизодов: {success_count}")
    print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
    s1 = success_count/num_episodes*100
    test_ac.append(s1)
    test_env =PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True, use_concentrtion = use_concentrtion, new_actions_rew = new_actions_rew, loop_penalty = 0.05)

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
            success_count += 1
    print(f"\nРезультаты за {num_episodes} эпизодов:")
    print(f"Успешных эпизодов: {success_count}")
    print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
    s1 = success_count/num_episodes*100
    train_ac.append(s1)
    fig, ax = plt.subplots()
    # plt.figure(figsize=(8, 5))
    ax.plot([i for i in range(len(test_ac))], test_ac)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy')
    fig.savefig(path_to_rep + "python3/fotos/test_ac" + cur_grp + ".png")
    plt.close(fig)
    fig, ax = plt.subplots()
    #plt.figure(figsize=(8, 5))
    plt.plot([i for i in range(len(train_ac))], train_ac)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    fig.savefig(path_to_rep + "python3/fotos/train_ac" + cur_grp + ".png")
    wandb.log({"test_ac_" + cur_grp: test_ac[-1], "train_ac_" + cur_grp: train_ac[-1]}, step= len(train_ac) - 1)
    plt.close()

class StatsCallback(BaseCallback):
    def __init__(self, stats_func, eval_freq=50000, verbose=0):
        super().__init__(verbose)
        self.stats_func = stats_func
        self.eval_freq = eval_freq
        self.last_eval_step = 0
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            if self.stats_func is not None:
                self.stats_func(self.model)


for group in ["all", "walls", "shapes", "base", "obstacles", "goals"]:
    wandb.init(project="test_" + group)
    print(group)
    cur_grp = group
    test_ac = []
    train_ac = []
    test_env = PushTargetEnv(f"/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/{group}/test", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True, augment = False, use_concentrtion = use_concentrtion, new_actions_rew = new_actions_rew, loop_penalty = 0.05)
    train_env = PushTargetEnv(f"/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/{group}/train", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True, augment = True, use_concentrtion = use_concentrtion, new_actions_rew = new_actions_rew, loop_penalty = 0.05)
    stats_callback = StatsCallback(stats_func=test_model)
    model = train_ppo(train_env, stats_callback, 1600000)

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
    wandb.finish()