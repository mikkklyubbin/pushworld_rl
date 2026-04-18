import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, DQN
from torch_geometric.nn import RGCNConv, global_mean_pool
from sb3_contrib import RecurrentPPO
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

class DoubleNets(nn.Module):
    def __init__(self, input_dim, output_dim, channels_in, channels_out):
        super(DoubleNets, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fl = nn.Flatten()

    def forward(self, x):
        x,y = x
        out1 = self.net1(x)
        out2 = self.net2(y)
        out2 = self.fl(out2)
        return torch.cat([out1, out2], dim=1)
    
    
def transform_flatten(x):
    return torch.cat([x[0], x[1].view(x[1].size(0), -1)], dim=1)
class PolicyForDifferentModalities(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,flatten_dim_in, flatten_dim_out, channels_in, channels_out, **kwargs):
        
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        self.action_net = DoubleNets(flatten_dim_in, flatten_dim_out, channels_in, channels_out)
        self.value_net = nn.Sequential(DoubleNets(flatten_dim_in, flatten_dim_out, channels_in, channels_out), nn.Linear(action_space.n, 1))
        
        
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi = self.action_net(features)  
        latent_vf = self.value_net(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        values = latent_vf.squeeze(-1)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf, _ = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        return latent_vf, log_prob, distribution.entropy().mean()
    
    




class MaskingPolicy(PolicyForDifferentModalities):
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
            if deterministic:
                actions = torch.argmax(log_prob, dim=1)
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
    