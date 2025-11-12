import sys
import os
import pushworld
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/home/mik/hse/Pushworld/pushworld-main/python3/src/pushworld'))
from stable_baselines3.common.evaluation import evaluate_policy
from pushworld.gym_env import PushWorldEnv
menv = PushWorldEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level1/A Tight Squeeze.pwp", 100)



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
            cell_shape = observation_space.shape
            sample_input = torch.rand(1, cell_shape[2], cell_shape[0], cell_shape[1])
            n_flatten = self.cnn(sample_input).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations):
        cell_obs = observations
        if len(cell_obs.shape) == 3:
            cell_obs = cell_obs.permute(2, 0, 1).unsqueeze(0)
        else:
            cell_obs = cell_obs.permute(0, 3, 1, 2)
        cell_features = self.cnn(cell_obs)
        return self.fc(cell_features)

def train_ppo(env):
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 128] 
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0002,
        n_epochs=2,
        ent_coef=0.01,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model.learn(total_timesteps=1000000)
    return model


model = train_ppo(menv)
model.save("ppo_custom_model")

# Загрузка модели (если нужно)
# model = PPO.load("ppo_custom_model")

# Оценка модели
mean_reward, std_reward = evaluate_policy(
    model, 
    menv, 
    n_eval_episodes=10,
    deterministic=True,
    render=False,   
    return_episode_rewards=True
)

print(f"Средняя награда: {mean_reward:.2f} +/- {std_reward:.2f}")