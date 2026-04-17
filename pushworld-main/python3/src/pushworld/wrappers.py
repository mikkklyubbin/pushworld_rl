import gym
import torch
import numpy as np
from pushworld.gym_env import PushTargetEnv

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, scale=1.0, shift=0.0, clip=None):
        super().__init__(env)
        self.scale = scale
        self.shift = shift
        self.clip = clip

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward = reward * self.scale + self.shift

        if self.clip is not None:
            reward = np.clip(reward, -self.clip, self.clip)

        return obs, reward, terminated, truncated, info
class Normalizer(PushTargetEnv):
    def __init__(self,*args, num_rollouts=1000, data = None, **kwargs):
        super().__init__(*args,**kwargs)
        assert isinstance(super().__self__, gym.Env)
        self.mean = 0
        self.std = 1
        self.reward_mean = 0
        self.reward_std = 1
        count = 0

        if data is not None:
            self.mean = data["mean"]
            self.std = data["std"]
            self.reward_mean = data["reward_mean"]
            self.reward_std = data["reward_std"]
        else:
            self.mean = {}
            self.std = {}
            for key in self.observation_space.spaces:
                self.mean[key] = np.zeros_like(self.observation_space.spaces[key].high, dtype=np.float32)
                self.std[key] = np.zeros_like(self.observation_space.spaces[key].high, dtype=np.float32)
            self.reward_mean = 0
            self.reward_std = 0
            for  _ in range(num_rollouts // 2):
                obs, _ = super().reset()
                terminated = False
                truncated = False
                while not terminated and not truncated:
                    action = super().action_space.sample()
                    while  obs["av"][action] == 0:
                        action = super().action_space.sample()
                    obs, reward, terminated, truncated, info = super().step(action)
                    for  key in self.observation_space.spaces:
                        if (key != "av"):
                            self.mean[key] += obs[key]
                    self.reward_mean += reward
                    count += 1
            for key in self.observation_space.spaces:
                self.mean[key] /= count
            self.reward_mean /= count
            count = 0
            for  _ in range(num_rollouts // 2):
                obs, _ = super().reset()
                terminated = False
                truncated = False
                while not terminated and not truncated:
                    action = super().action_space.sample()
                    while  obs["av"][action] == 0:
                        action = super().action_space.sample()
                    obs, reward, terminated, truncated, info = super().step(action)
                    for  key in self.observation_space.spaces:
                        if (key != "av"):
                            self.std[key] += (obs[key] - self.mean[key]) ** 2
                    self.reward_std += (reward - self.reward_mean) ** 2
                    count += 1

            for key in self.observation_space.spaces:
                self.std[key] /= count
                self.std[key] = np.sqrt(self.std[key])
            self.reward_std /= count
            self.reward_std = np.sqrt(self.reward_std)
        print("reward_mean", self.reward_mean) 
        print("reward_std", self.reward_std)
        print("predicted full", (10 - self.reward_mean) / self.reward_std)

    def tranform_obs(self, obs):
        # for key in self.observation_space.spaces:
        #     normal = self.std[key] > 1e-7
        #     obs[key] = (obs[key] - self.mean[key]) / (self.std[key] + 1e-8) * normal + obs[key] * (1 - normal)
        return obs

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = self.tranform_obs(obs)
        if  not  obs in self.observation_space:
            print("obs", obs)
            print("obs space", self.observation_space)
        assert obs in self.observation_space
        return obs,info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.tranform_obs(obs)
        assert obs in self.observation_space
        if self.reward_std > 1e-7:
            reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return obs, reward, terminated, truncated, info
    
    def export(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
        }
    
    