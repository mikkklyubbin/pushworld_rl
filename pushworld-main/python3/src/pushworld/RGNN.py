import sys
import os
import pushworld
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from pushworld.gym_env import PushWorldEnv
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/home/mik/hse/Pushworld/pushworld-main/python3/src/pushworld'))

menv = PushWorldEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/all/train", 100, need_pddl=True)
model_save_path = "/home/mik/hse/Pushworld/pushworld-main/python3/model/bst"
eval_env = DummyVecEnv([lambda: PushWorldEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/all/train", 100)])

eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=model_save_path,
    eval_freq=200000,
    n_eval_episodes=10, 
    deterministic=True,
    render=False,
    verbose=1
)
class ResidualRGCN(nn.Module):
    """
    Residual RGCN слой с Batch Normalization
    """
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.conv = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.batch_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_type):
        residual = x
        x = self.conv(x, edge_index, edge_type)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, num_relations=8):
        self.is_multimodal =  hasattr(observation_space, 'spaces')
        print(type(observation_space))
        print(self.is_multimodal)
        if self.is_multimodal:
            super(CustomCNN, self).__init__(observation_space, features_dim)
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((6, 6)), 
                nn.Flatten(),
            )
            with torch.no_grad():
                sample_input = torch.rand(1, 3, *observation_space['cell'].shape[:2])
                n_flatten = self.cnn(sample_input).shape[1]
            
            self.cnn_fc = nn.Sequential(
                nn.Linear(n_flatten, 256),
                nn.ReLU(),
                nn.Linear(256, features_dim // 2),
                nn.ReLU(),
            )
            
            max_nodes = observation_space['graph'].high[0][0] + 1 
            print(max_nodes)
            self.node_embedding = nn.Embedding(max_nodes, 64)
            self.rgcn_layers = self._build_rgcn_layers(64, num_relations, 3) 
            
            self.graph_processor = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, features_dim // 2),
                nn.ReLU(),
            )

            self.fusion = nn.Sequential(
                nn.Linear(features_dim, features_dim),
                nn.ReLU(),
            )
        else:
            super(CustomCNN, self).__init__(observation_space, features_dim)
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((6, 6)), 
                nn.Flatten(),
            )
            
            with torch.no_grad():
                print(observation_space)
                sample_input = torch.rand(1, 3, *observation_space.shape[:2])
                n_flatten = self.cnn(sample_input).shape[1]
            
            self.cnn_fc = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )
    
    def _build_rgcn_layers(self, hidden_dim, num_relations, num_layers):
        layers = nn.ModuleList()
        
        layers.append(nn.Sequential(
            RGCNConv(hidden_dim, hidden_dim, num_relations),
            nn.ReLU()
        ))
        
        for _ in range(num_layers - 2):
            layers.append(ResidualRGCN(hidden_dim, num_relations))
        
        layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
        
        return layers
        
    def forward(self, observations):
        if self.is_multimodal:
            cell_obs = observations['cell']
            if len(cell_obs.shape) == 3:
                cell_obs = cell_obs.permute(2, 0, 1).unsqueeze(0)
            elif len(cell_obs.shape) == 4:
                cell_obs = cell_obs.permute(0, 3, 1, 2)
            
            cnn_features = self.cnn(cell_obs)
            cnn_features = self.cnn_fc(cnn_features)
            gr_obs = observations['graph']
            if len(gr_obs.shape) == 3:
                gr_obs = gr_obs.unsqueeze(0)

            graph_features = self._process_graph(gr_obs)
            
            combined = torch.cat([cnn_features, graph_features], dim=1)
            return self.fusion(combined)
        else:
            cell_obs = observations
            if len(cell_obs.shape) == 3:
                cell_obs = cell_obs.permute(2, 0, 1).unsqueeze(0)
            elif len(cell_obs.shape) == 4: 
                cell_obs = cell_obs.permute(0, 3, 1, 2)
            
            cnn_features = self.cnn(cell_obs)
            return self.cnn_fc(cnn_features)
    
    def _process_graph(self, graph_obs):
        """Обработка графовых наблюдений в матричной форме"""
        batch_size = 1
        graph_features = []

        for batch_idx in range(graph_obs.shape[0]):
            edges, edge_types = [], []

            for i in range(graph_obs.shape[1]):
                source, target, edge_type = graph_obs[batch_idx][i]
                if source != 0 or target != 0 or edge_type != 0:
                    edges.append((source, target))
                    edge_types.append(int(edge_type))

            if not edges:
                graph_embedding = torch.zeros(self.features_dim // 2)
            else:
                edges = torch.tensor(edges)
                edge_index = edges.t().long().to("cuda")
                edge_type = torch.tensor(edge_types, dtype=torch.long)

                unique_nodes = torch.unique(edge_index).long().to("cuda")
                x = self.node_embedding(unique_nodes)
                for layer in self.rgcn_layers:
                    if isinstance(layer, nn.Sequential):
                        x = layer[0](x, edge_index, edge_type)
                        x = layer[1](x)
                    elif isinstance(layer, ResidualRGCN):
                        x = layer(x, edge_index, edge_type)
                    else:
                        x = layer(x, edge_index, edge_type)

                graph_embedding = x.mean(dim=0)
                graph_embedding = self.graph_processor(graph_embedding.unsqueeze(0)).squeeze(0)

            graph_features.append(graph_embedding)
        return torch.stack(graph_features)

def train_ppo(env):
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[64, 64]
    )
    
    model = PPO(
        "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0002,
        n_epochs=2,
        ent_coef=0.01,
        batch_size = 2,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("Начинаем обучение...")
    model.learn(total_timesteps=500000, callback = eval_callback)
    return model


model = train_ppo(menv)
model.save("/home/mik/hse/Pushworld/pushworld-main/python3/model/ppo_graph")


print("Оценка модели...")
mean_reward, std_reward = evaluate_policy(
    model, 
    menv, 
    n_eval_episodes=5,
    deterministic=True
)

print(f"Средняя награда: {mean_reward:.2f} +/- {std_reward:.2f}")
    