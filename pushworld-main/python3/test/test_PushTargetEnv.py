import sys
import os
import pushworld
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
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
from pushworld.callbacks import StatsCallback, MetricsCallback
from pushworld.gym_env import INFORMATION_CHANEL_PER_OBJECT, INFORMATION_CHANEL_STATIC
from pushworld.eval import eval_ac
path_to_rep = "/home/mik/hse/Pushworld/pushworld-main/"
use_concentrtion:bool = False
new_actions_rew:float = 0
block_rew:float = 0
block_peny:float = 0
use_block =  False
loop_penalty = 0.05
rgb = False
need_pddl = False
use_MDP = True
print("sss", rgb)
config = {
    "use_concentrtion": use_concentrtion,
    "new_actions_rew": new_actions_rew,
    "block_rew": block_rew,
    "block_peny": block_peny, 
    "use_block":use_block,
    "loop_penalty":loop_penalty,
    "need_pddl":need_pddl,
    "to_height":11,
    "to_width":11,
    "max_obj":5,
    "rgb": rgb,
    "use_MDP": use_MDP,
}
print(rgb)
in_channels = 3
if not rgb:
    in_channels = 5 * INFORMATION_CHANEL_PER_OBJECT + INFORMATION_CHANEL_STATIC # change 5 to max obj locally
print("in_channels", in_channels)
model_kwargs = {"in_channels": in_channels}

config_train = {"node_feature": 64, "features_dim": 512, "hidden_dim": 512, "batch_size": 128, "n_epochs": 2, "need_pddl":need_pddl,}
print(rgb)
menv = PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100, augment = True, **config)


eval_env =  PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/test", 100, **config)
name_of_test = "test_learning_NOCON_AUGMENT_NEWACT_LOOPPEN"
wandb.init(project="test_", config={**config_train, **config},name="check_small_cnn+rgcn")
model_save_path = path_to_rep + "python3/model/bst2"

test_ac = []
train_ac = []

def test_model(model):
    test_env = PushTargetEnv(path_to_rep + f"benchmark/puzzles/level0/all/test", 100,  **config)

    num_episodes = 200
    s1 = eval_ac(test_env, num_episodes, model, verbose=True)
    test_ac.append(s1)
    test_env =PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100, seq = True, **config)
    num_episodes = 200
    s1 = eval_ac(test_env, num_episodes, model, verbose=True)
    train_ac.append(s1)
    fig, ax = plt.subplots()
    # plt.figure(figsize=(8, 5))
    ax.plot([i for i in range(len(test_ac))], test_ac)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy')
    fig.savefig(path_to_rep + "python3/fotos/test_ac.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    #plt.figure(figsize=(8, 5))
    plt.plot([i for i in range(len(train_ac))], train_ac)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    fig.savefig(path_to_rep + "python3/fotos/train_ac.png")
    wandb.log({"test_ac": test_ac[-1], "train_ac": train_ac[-1]})
    plt.close(fig)
    
             

eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=model_save_path,
    eval_freq=50000,
    n_eval_episodes=10, 
    deterministic=False,
    render=False,
    verbose=1
)

stats_callback = StatsCallback(stats_func=test_model)
metric_call = MetricsCallback(1)

combined_callback = CallbackList([eval_callback, stats_callback, metric_call])

print(menv.observation_space)
model = train_ppo(menv, combined_callback, **config_train, model_kwargs=model_kwargs)

model.save(path_to_rep + "python3/model/ppo_custom_model")

