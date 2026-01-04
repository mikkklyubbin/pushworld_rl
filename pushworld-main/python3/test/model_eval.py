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
import matplotlib.pyplot as plt
import numpy as np
import dataframe_image as dfi
import cv2
import pandas as pd
from pushworld.gym_env import savergb
model_save_path = "/home/mik/hse/Pushworld/pushworld-main/python3/model/bst2/best_model.zip"

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)), 
            nn.Flatten(),
        )
        
        with torch.no_grad():
            cell_shape = observation_space.spaces['cell'].shape
            sample_input = torch.rand(1, cell_shape[2], cell_shape[0], cell_shape[1])
            n_flatten = self.cnn(sample_input).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(n_flatten + observation_space.spaces['positions'].shape[0] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations):
        cell_obs = observations['cell']
        if len(cell_obs.shape) == 3:
            cell_obs = cell_obs.permute(2, 0, 1).unsqueeze(0)
        else:
            cell_obs = cell_obs.permute(0, 3, 1, 2)
        cell_features = self.cnn(cell_obs)
        pos_obs = observations['positions']
        batch_size = pos_obs.shape[0]
        pos_features = pos_obs.reshape(batch_size, -1)
        
        combined = torch.cat([cell_features, pos_features], dim=1)
        return self.fc(combined)



class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _predict(self, observation, deterministic=False):
        """
        Override _predict to ensure proper observation handling
        """
        if not isinstance(observation, dict):
            observation = self.obs_to_tensor(observation)
        else:
            observation = {key: torch.as_tensor(value, device=self.device) 
                          for key, value in observation.items()}
        
        with torch.no_grad():

            actions, values, log_prob = self.forward(observation, deterministic=deterministic)

        return actions
    
    def forward(self, obs, deterministic=False):
        #print(obs["cell"].shape)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
    
        action_mask_data = obs["av"]
        if isinstance(action_mask_data, np.ndarray):
            action_mask = torch.tensor(action_mask_data, dtype=torch.float32, 
                                      device=distribution.distribution.logits.device)
        else:
            action_mask = action_mask_data.to(dtype=torch.float32, 
                                            device=distribution.distribution.logits.device)
        
        if len(action_mask.shape) == 1:
            action_mask = action_mask.unsqueeze(0) 
        
        modified_logits = distribution.distribution.logits.clone()
        modified_logits = modified_logits - (1 - action_mask) * 1e9
        
        distribution.distribution = torch.distributions.Categorical(logits=modified_logits)
        
        values = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_mask_data = obs["av"]
        if isinstance(action_mask_data, np.ndarray):
            action_mask = torch.tensor(action_mask_data, dtype=torch.float32, 
                                      device=distribution.distribution.logits.device)
        else:
            action_mask = action_mask_data.to(dtype=torch.float32, 
                                            device=distribution.distribution.logits.device)
        
        if len(action_mask.shape) == 1:
            action_mask = action_mask.unsqueeze(0)
            
        modified_logits = distribution.distribution.logits.clone()
        modified_logits = modified_logits - (1 - action_mask) * 1e9
        distribution.distribution = torch.distributions.Categorical(logits=modified_logits)

        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy
    

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[128, 128]
)

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

id:int = 0
cool_table = pd.DataFrame({'Type':[], 'Test%':[], 'Train%':[]})
#, "walls", "shapes", "base", "obstacles", "goals"
for group in ["all", "walls", "shapes", "base", "obstacles", "goals"]:
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
            savergb(rgb, "/home/mik/hse/Pushworld/pushworld-main/python3/fotos/" + str(episode) + ".jpg")
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