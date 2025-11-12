import sys
import os
import pushworld
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/home/mik/hse/Pushworld/pushworld-main/python3/src/pushworld'))
from stable_baselines3.common.evaluation import evaluate_policy
from pushworld.gym_env import PushTargetEnv
menv = PushTargetEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/all/train", 100)

eval_env = DummyVecEnv([lambda: PushTargetEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/all/train", 100)])

# Определение пути для сохранения лучшей модели
model_save_path = "/home/mik/hse/Pushworld/pushworld-main/python3/model/bst"

# Создание EvalCallback
eval_callback = EvalCallback(
    eval_env, # Оценочная среда
    best_model_save_path=model_save_path, # Путь для сохранения
    eval_freq=200000, # Частота оценки (каждые 10000 шагов)
    n_eval_episodes=10, # Количество эпизодов для оценки
    deterministic=True,
    render=False,
    verbose=1
)

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
    
    model.learn(total_timesteps=30000000, callback=eval_callback)
    return model


model = train_ppo(menv)
model.save("/home/mik/hse/Pushworld/pushworld-main/python3/model/ppo_custom_model")

# test_env = PushTargetEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/all/train", 100)

# # Загружаем обученную модель
# model = PPO.load("ppo_custom_model")

# # Тестируем модель
# num_episodes = 100
# success_count = 0

# for episode in range(num_episodes):
#     obs = test_env.reset()
#     done = False
#     episode_rewards = []
    
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = test_env.step(action)
#         episode_rewards.append(reward)
    
#     # Проверяем, завершился ли эпизод успехом (terminated=True)
#     if info.get('terminated', False):
#         success_count += 1
        
#     print(f"Эпизод {episode + 1}: Награда = {sum(episode_rewards):.2f}, Успех = {info.get('terminated', False)}")

# print(f"\nРезультаты за {num_episodes} эпизодов:")
# print(f"Успешных эпизодов: {success_count}")
# print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
