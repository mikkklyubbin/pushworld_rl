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
use_concentrtion:bool = False
new_actions_rew:float = 0
block_rew:float = 0
block_peny:float = 0
use_block =  False
loop_penalty = 0.05
rgb = False
need_pddl = False
use_MDP = False
use_DIRECT = True
max_obj = 5

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
if not rgb:
    in_channels = max_obj * INFORMATION_CHANEL_PER_OBJECT + INFORMATION_CHANEL_STATIC # change 5 to max obj locally
print("in_channels", in_channels)
model_kwargs = {"in_channels": in_channels}

config_train = {"node_feature": 64, "features_dim": 512, "hidden_dim": 512, "need_pddl":need_pddl, **model_kwargs}
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=config_train,
    net_arch=dict(pi=[512, 256], vf=[512, 256])
)

model_save_path = "/home/mik/hse/Pushworld/pushworld-main/python3/model/bst2/best_model.zip"
model = PPO.load(
    model_save_path,
    custom_objects={
        "policy_class": CustomPolicy,
        "policy_kwargs": policy_kwargs
    },
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
model_kwargs = {"in_channels": in_channels}

config_train = {"node_feature": 64, "features_dim": 512, "hidden_dim": 512, "batch_size": 128, "n_epochs": 2, "need_pddl":need_pddl,}
path_to_rep = "/home/mik/hse/Pushworld/pushworld-main/"
for group in ["all"]:
    print(group)
    test_env = PushTargetEnv(path_to_rep + f"benchmark/puzzles/level0/all/test", 100,seq=True, **config)
    num_episodes = 200
    success_count = 0
    for episode in range(num_episodes):

        obs, _ = test_env.reset()  
        terminated = False
        truncated = False
        episode_rewards = []
        rgb_ar  = [test_env.render()]
        while not terminated:
            action, _ = model.predict(obs)  

            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_rewards.append(reward)
            rgb_ar.append(test_env.render())
            if (truncated):
                break
            
        if terminated:
            # savergb(rgb, "/home/mik/hse/Pushworld/pushworld-main/python3/fotos/" + str(episode) + ".jpg")
            success_count += 1
        #savergb(test_env.render(), "/home/mik/hse/Pushworld/pushworld-main/python3/1.jpg")
        res = test_env.render_acts()
        create_rgb_video_opencv(res, "/home/mik/hse/Pushworld/pushworld-main/python3/fotos/check" + str(episode % 10) + ".avi")
    print(f"\nРезультаты за {num_episodes} эпизодов:")
    print(f"Успешных эпизодов: {success_count}")
    print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
    s1 = success_count/num_episodes*100
