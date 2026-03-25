import sys
import os
import pushworld
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from stable_baselines3.common.callbacks import EvalCallback
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/home/mik/hse/Pushworld/pushworld-main/python3/src/pushworld'))
from pushworld.gym_env import PushTargetEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from pushworld.model import CustomCNN, CustomPolicy, train_ppo, train_rec_PPO, CustomRecurrentPolicy
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
from pushworld.callbacks import StatsCallback, MetricsCallback
from pushworld.gym_env import INFORMATION_CHANEL_PER_OBJECT, INFORMATION_CHANEL_STATIC
from pushworld.eval import eval_ac, eval_ac_rec
from pushworld.load_model import load_PPO_REC
import comet_ml  
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
use_DIRECT = False
max_obj = 5
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
    "max_obj":max_obj,
    "rgb": rgb,
    "use_MDP": use_MDP,
    "use_DIRECT": use_DIRECT,
}
print(rgb)
in_channels = 3
menv = PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100, augment = True, **config)
if not rgb:
    in_channels = menv.all_chanells
print("in_channels", in_channels)
model_kwargs = {"in_channels": in_channels}
model = load_PPO_REC("/home/mik/hse/Pushworld/pushworld-main/python3/model/ppo_custom_model.zip", config=config)
test_env = PushTargetEnv(path_to_rep + f"benchmark/puzzles/level0/all/test", 100, seq =True,  **config)
num_episodes = 200
s1 = eval_ac_rec(test_env, num_episodes, model, verbose=True)