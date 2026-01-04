import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
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
    
def train_ppo(env, callback, total_timesteps=60000000):

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 128]
    )

    model = PPO(
        CustomPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0002,
        n_epochs=2,
        ent_coef=0.01,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model