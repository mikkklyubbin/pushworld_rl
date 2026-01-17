import sys
import os
import pushworld
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/home/mik/hse/Pushworld/pushworld-main/python3/src/pushworld'))
from stable_baselines3.common.evaluation import evaluate_policy
from pushworld.gym_env import PushTargetEnv
from pushworld.rendering import savergb
menv = PushTargetEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/all/test/level_0_all_test_0.pwp", 100)

menv.reset()

menv.step(7)
savergb(menv.render(), "2.jpg")