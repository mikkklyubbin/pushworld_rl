import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, DQN
from torch_geometric.nn import RGCNConv, global_mean_pool
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, need_pddl  = False, node_feature = 64, hidden_dim = 512, in_channels = 3):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        print(in_channels)
        self.need_pddl = need_pddl
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((6, 6)), 
            nn.Flatten(),
        )
        
        
        with torch.no_grad():
            cell_shape = observation_space.spaces['cell'].shape
            print("cell_shape", cell_shape)
            sample_input = torch.rand(1, cell_shape[2], cell_shape[0], cell_shape[1])
            n_flatten = self.cnn(sample_input).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(n_flatten + observation_space.spaces['positions'].shape[0] * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.for_last = nn.Sequential(
            nn.Linear(hidden_dim + observation_space.spaces['last_ac'].shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )
        if (self.need_pddl):
            self.max_nodes = observation_space["edges"].high[0][0] + 1
            self.node_feat_dim = node_feature
            self.num_relations = int(observation_space["types"].high[0]) + 1
            self.rgcn1 = RGCNConv(self.node_feat_dim, 64, num_relations=self.num_relations)
            self.rgcn2 = RGCNConv(64, 64, num_relations=self.num_relations)
            self.node_embeddings = nn.Embedding(
                self.max_nodes,
                self.node_feat_dim 
            )
            self.node_processor = nn.Linear(64, 64)
            self.graph_projection = nn.Linear(64, features_dim)
            self.combiner = nn.Sequential(
                nn.Linear(hidden_dim + features_dim, hidden_dim),
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
        res = self.fc(combined)
        if (self.need_pddl):
            edge_index = observations["edges"]
            edge_type = observations["types"]
            batch_size = edge_index.shape[0]
            node_indices = torch.arange(
                self.max_nodes, 
                device=edge_index.device
            ).repeat(batch_size, 1)
            x = self.node_embeddings(node_indices)
            x = x.view(-1, self.node_feat_dim)
            edge_index = edge_index.view(2, -1).long()
            edge_type = edge_type.view(-1).long()
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(self.max_nodes)
            x = self.rgcn1(x, edge_index, edge_type)
            x = torch.relu(x)
            x = self.rgcn2(x, edge_index, edge_type)
            x = torch.relu(x)
            x = self.node_processor(x)
            graph_embedding = global_mean_pool(x, batch)
            x = self.graph_projection(graph_embedding)
            res = res + self.combiner(torch.cat([res, x], dim=1))

        combined2 = torch.cat([res, observations['last_ac']], dim=1)
        return self.for_last(combined2) + res



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
    
def train_ppo(env, callback, total_timesteps=60000000, need_pddl = False, node_feature = 64, features_dim=512, hidden_dim=512, batch_size = 128, n_epochs=2, model_kwargs = {"in_channels": 3}):

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=features_dim, need_pddl = need_pddl, node_feature = node_feature, hidden_dim=hidden_dim, **model_kwargs),
        net_arch=dict(pi=[512, 256], vf=[512, 256])
    )

    model = PPO(
        CustomPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0002,
        n_epochs=n_epochs,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        batch_size=batch_size,
        vf_coef=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model

def train_dqn(env, callback, total_timesteps=60000000, need_pddl = False, node_feature = 64, features_dim=512, hidden_dim=512, batch_size = 128, n_epochs=2, model_kwargs = {"in_channels": 3}):
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            need_pddl=need_pddl,
            node_feature=node_feature,
            hidden_dim=hidden_dim,
            **model_kwargs
        ),
        net_arch=[512, 256]
    )
    model = DQN(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=10000,
        batch_size=batch_size,
        tau=1.0, 
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.learn(total_timesteps=total_timesteps, callback= callback)
    return model
