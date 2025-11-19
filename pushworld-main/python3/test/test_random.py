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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import matplotlib.pyplot as plt
import numpy as np
import cv2
path_to_rep = "/home/mikk/PushWorld/pushworld_rl/pushworld-main/"
test_env = PushTargetEnv(path_to_rep + f"benchmark/puzzles/level0/all/test", 100, to_height = 11, to_width = 11, max_obj = 5, seq = True)

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

num_episodes = 200
success_count = 0
for episode in range(num_episodes):
    obs, _ = test_env.reset()  
    terminated = False
    truncated = False
    episode_rewards = []
    while not terminated:
        action = test_env.action_space.sample()
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_rewards.append(reward)
        if (truncated):
            break
    if terminated:
        print(1)
        print(episode)
        create_rgb_video_opencv(test_env.render_video(),path_to_rep + "python3/fotos/" + str(episode) + ".avi")
        rgb = test_env.render()
        success_count += 1
print(f"\nРезультаты за {num_episodes} эпизодов:")
print(f"Успешных эпизодов: {success_count}")
print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
s1 = success_count/num_episodes*100
