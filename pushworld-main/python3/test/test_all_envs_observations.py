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
path_to_rep = "/home/mik/hse/Pushworld/pushworld-main/"
use_concentrtion:bool = False
new_actions_rew:float = 0
use_block = False
block_rew = 0.005
block_peny = 0.1
need_pddl = False
rgb = False



menv = PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/test/level_0_all_test_199.pwp", 100, augment = False, use_concentrtion = use_concentrtion, new_actions_rew = new_actions_rew, loop_penalty = 0.05, block_rew = block_rew, block_peny = block_peny, use_block = use_block, need_pddl = need_pddl, rgb = rgb,use_MDP = True, use_DIRECT = False, max_obj = 5)


name_of_test = "test_learning_NOCON_AUGMENT_NEWACT_LOOPPEN"

o, i = menv.reset()
rgb = menv.render()

print(o["av"])
savergb(rgb, "/home/mik/hse/Pushworld/pushworld-main/python3/1.jpg")
while (True):
    a = int(input())
    o, r, ter, trun, info = menv.step(a)
    assert(o in menv._observation_space)
    rgb = menv.render()
    print(o['av'])







