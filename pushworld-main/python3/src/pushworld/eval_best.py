from pushworld.model import CustomCNN, CustomPolicy
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from pushworld.gym_env import PushTargetEnv
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[128, 128]
)
model_save_path = "/home/mik/hse/Pushworld/pushworld-main/python3/model/bst2/best_model.zip"
model = PPO.load(
    model_save_path,
    custom_objects={
        "policy_class": CustomPolicy,
        "policy_kwargs": {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": dict(features_dim=128),
            "net_arch": [128, 128]
        }
    },
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
for group in ["all"]:
    print(group)
    test_env = PushTargetEnv(f"/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/{group}/test", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True)

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
