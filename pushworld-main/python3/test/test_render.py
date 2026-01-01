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
from pushworld.gym_env import PushTargetEnv, savergb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import matplotlib.pyplot as plt
import numpy as np
import cv2
path_to_rep = "/home/mikk/PushWorld/pushworld_rl/pushworld-main/"
menv = PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100)

eval_env =  PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/test", 100)
eval_env.reset()

savergb(eval_env.render(), "/home/mikk/PushWorld/pushworld_rl/pushworld-main/python3/image.jpg")