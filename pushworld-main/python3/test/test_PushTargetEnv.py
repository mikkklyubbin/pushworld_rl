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
from pushworld.model import CustomCNN, CustomPolicy, train_ppo
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pushworld.rendering import savergb, create_rgb_video_opencv
path_to_rep = "/home/mik/hse/Pushworld/pushworld-main/"
menv = PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100)

eval_env =  PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/test", 100)

model_save_path = path_to_rep + "python3/model/bst2"

test_ac = []
train_ac = []

def test_model(model):
    test_env = PushTargetEnv(path_to_rep + f"benchmark/puzzles/level0/all/test", 50, to_height = 11, to_width = 11, max_obj = 5, seq = True)

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
    test_env =PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 50, to_height = 11, to_width = 11, max_obj = 5, seq = True)

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
    plt.figure(figsize=(8, 5))
    plt.plot([i for i in range(len(test_ac))], test_ac)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.savefig(path_to_rep + "python3/fotos/test_ac.png")
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot([i for i in range(len(train_ac))], train_ac)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.savefig(path_to_rep + "python3/fotos/train_ac.png")
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

eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=model_save_path,
    eval_freq=10000,
    n_eval_episodes=10, 
    deterministic=False,
    render=False,
    verbose=1
)

stats_callback = StatsCallback(stats_func=test_model)

combined_callback = CallbackList([eval_callback, stats_callback])


model = train_ppo(menv, combined_callback)

model.save(path_to_rep + "python3/model/ppo_custom_model")

