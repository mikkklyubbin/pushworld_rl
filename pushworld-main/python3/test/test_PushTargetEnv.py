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
import cv2
path_to_rep = "/home/mik/hse/Pushworld/pushworld-main/"
menv = PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100)

eval_env =  PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/test", 100)

model_save_path = path_to_rep + "python3/model/bst2"

test_ac = []
train_ac = []

def create_rgb_video_opencv(data, output_file='rgb_video.avi', fps=10):
    """
    Создает видео из RGB данных используя OpenCV
    """
    # Получаем размеры первого кадра
    first_frame = data[0]
    height, width = first_frame.shape[:2]
    
    # Создаем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for i, rgb_frame in enumerate(data):
        # Конвертируем RGB в BGR (OpenCV использует BGR)
        bgr_frame = cv2.cvtColor((rgb_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Добавляем текст с номером кадра
        cv2.putText(bgr_frame, f'Frame: {i}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(bgr_frame)
    
    out.release()
    print(f"Video saved as {output_file}")

def test_model(model):
    test_env = PushTargetEnv(path_to_rep + f"benchmark/puzzles/level0/all/test", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True)

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
            success_count += 1
    print(f"\nРезультаты за {num_episodes} эпизодов:")
    print(f"Успешных эпизодов: {success_count}")
    print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
    s1 = success_count/num_episodes*100
    test_ac.append(s1)
    test_env =PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True)

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
            success_count += 1
    print(f"\nРезультаты за {num_episodes} эпизодов:")
    print(f"Успешных эпизодов: {success_count}")
    print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
    s1 = success_count/num_episodes*100
    train_ac.append(s1)
    plt.figure(figsize=(8, 5))
    plt.plot([i for i in range(len(test_ac))], test_ac)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.savefig(path_to_rep + "python3/fotos/test_ac.png")
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot([i for i in range(len(train_ac))], train_ac)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.savefig(path_to_rep + "python3/fotos/train_ac.png")
    plt.close()
    


class StatsCallback(BaseCallback):
    def __init__(self, stats_func, eval_freq=50000, verbose=0):
        super().__init__(verbose)
        self.stats_func = stats_func
        self.eval_freq = eval_freq
        self.last_eval_step = 0
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Вызывается в конце rollout"""
        # Проверяем, прошло ли достаточно шагов с последнего вызова
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            if self.stats_func is not None:
                self.stats_func(self.model)

# Комбинируем оба callback'а
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=model_save_path,
    eval_freq=10000,
    n_eval_episodes=10, 
    deterministic=True,
    render=False,
    verbose=1
)

stats_callback = StatsCallback(stats_func=test_model)

# Объединяем в один callback
combined_callback = CallbackList([eval_callback, stats_callback])

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
        # Убедитесь, что observation является тензором
        if not isinstance(observation, dict):
            observation = self.obs_to_tensor(observation)
        else:
            # Если это уже словарь тензоров, убедитесь они на правильном устройстве
            observation = {key: torch.as_tensor(value, device=self.device) 
                          for key, value in observation.items()}
        
        with torch.no_grad():
            # forward возвращает кортеж (actions, values, log_prob)
            actions, values, log_prob = self.forward(observation, deterministic=deterministic)
        # Извлекаем только actions и применяем .cpu().numpy() к ним
        return actions
    
    def forward(self, obs, deterministic=False):
        #print(obs["cell"].shape)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
    
        # Исправленное получение маски с поддержкой batch
        action_mask_data = obs["av"]
        if isinstance(action_mask_data, np.ndarray):
            action_mask = torch.tensor(action_mask_data, dtype=torch.float32, 
                                      device=distribution.distribution.logits.device)
        else:
            # Если это уже тензор
            action_mask = action_mask_data.to(dtype=torch.float32, 
                                            device=distribution.distribution.logits.device)
        
        # Убедимся, что маска имеет правильную shape
        if len(action_mask.shape) == 1:
            action_mask = action_mask.unsqueeze(0)  # Добавляем batch dimension
        
        # Правильное применение маски
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

        # То же исправление для evaluate_actions
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
    

def train_ppo(env):

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 128]
    )

    model = PPO(
        CustomPolicy,  # Используем кастомную политику
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0002,
        n_epochs=2,
        ent_coef=0.01,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    model.learn(total_timesteps=60000000, callback=combined_callback)
    return model


model = train_ppo(menv)

model.save(path_to_rep + "python3/model/ppo_custom_model")

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
