from pushworld.model import CustomCNN, CustomPolicy
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from pushworld.gym_env import PushTargetEnv, INFORMATION_CHANEL_PER_OBJECT, INFORMATION_CHANEL_STATIC
from rendering import create_rgb_video_opencv, savergb
def load_PPO_model(model_save_path):
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

    config_train = {"node_feature": 64, "features_dim": 512, "hidden_dim": 512, "need_pddl":need_pddl, **model_kwargs}
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=config_train,
        net_arch=dict(pi=[512, 256], vf=[512, 256])
    )
    model = PPO.load(
        model_save_path,
        custom_objects={
            "policy_class": CustomPolicy,
            "policy_kwargs": policy_kwargs
        },
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return model