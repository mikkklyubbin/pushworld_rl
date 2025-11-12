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

# Убедитесь, что путь правильный
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '/home/mik/hse/Pushworld/pushworld-main/python3/src/pushworld'))

# Создаем среду
menv = PushWorldEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level1/A Tight Squeeze.pwp", 100, need_pddl=True)

class ResidualRGCN(nn.Module):
    """
    Residual RGCN слой с Batch Normalization
    """
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.conv = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index, edge_type):
        residual = x
        x = self.conv(x, edge_index, edge_type)
        x = self.batch_norm(x)
        x = F.relu(x + residual)  # Добавляем residual connection
        return x

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, num_relations=8):
        # Проверяем тип observation_space
        self.is_multimodal = isinstance(observation_space, gym.spaces.Dict)
        
        if self.is_multimodal:
            # Для многомодальных наблюдений
            super(CustomCNN, self).__init__(observation_space, features_dim)
            
            # CNN для изображений
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((6, 6)), 
                nn.Flatten(),
            )
            
            # Определяем размер выхода CNN
            with torch.no_grad():
                sample_input = torch.rand(1, 3, *observation_space['cell'].shape[:2])
                n_flatten = self.cnn(sample_input).shape[1]
            
            self.cnn_fc = nn.Sequential(
                nn.Linear(n_flatten, 256),
                nn.ReLU(),
                nn.Linear(256, features_dim // 2),
                nn.ReLU(),
            )
            
            # Графовая часть
            max_nodes = 100  # Установите подходящее максимальное количество узлов
            self.node_embedding = nn.Embedding(max_nodes, 64)
            self.rgcn_layers = self._build_rgcn_layers(64, num_relations, 3)  # Уменьшили количество слоев
            
            self.graph_processor = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, features_dim // 2),
                nn.ReLU(),
            )

            self.fusion = nn.Sequential(
                nn.Linear(features_dim, features_dim),
                nn.BatchNorm1d(features_dim),
                nn.ReLU(),
            )
        else:
            # Для обычных наблюдений
            super(CustomCNN, self).__init__(observation_space, features_dim)
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((6, 6)), 
                nn.Flatten(),
            )
            
            with torch.no_grad():
                sample_input = torch.rand(1, 3, *observation_space.shape[:2])
                n_flatten = self.cnn(sample_input).shape[1]
            
            self.cnn_fc = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )
    
    def _build_rgcn_layers(self, hidden_dim, num_relations, num_layers):
        layers = nn.ModuleList()
        
        # Первый слой
        layers.append(nn.Sequential(
            RGCNConv(hidden_dim, hidden_dim, num_relations),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ))
        
        # Промежуточные слои
        for _ in range(num_layers - 2):
            layers.append(ResidualRGCN(hidden_dim, num_relations))
        
        # Последний слой
        layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
        
        return layers
        
    def forward(self, observations):
        if self.is_multimodal:
            # Обработка изображений
            cell_obs = observations['cell']
            if len(cell_obs.shape) == 3:  # (H, W, C)
                cell_obs = cell_obs.permute(2, 0, 1).unsqueeze(0)
            elif len(cell_obs.shape) == 4:  # (B, H, W, C)
                cell_obs = cell_obs.permute(0, 3, 1, 2)
            
            cnn_features = self.cnn(cell_obs)
            cnn_features = self.cnn_fc(cnn_features)
            
            # Обработка графов
            graph_features = self._process_graph(observations['graph'])
            
            # Объединение признаков
            combined = torch.cat([cnn_features, graph_features], dim=1)
            return self.fusion(combined)
        else:
            # Обработка только изображений
            cell_obs = observations
            if len(cell_obs.shape) == 3:  # (H, W, C)
                cell_obs = cell_obs.permute(2, 0, 1).unsqueeze(0)
            elif len(cell_obs.shape) == 4:  # (B, H, W, C)
                cell_obs = cell_obs.permute(0, 3, 1, 2)
            
            cnn_features = self.cnn(cell_obs)
            return self.cnn_fc(cnn_features)
    
    def _process_graph(self, graph_obs):
    """Обработка графовых наблюдений в матричной форме"""
    # graph_obs имеет форму [max_edges, 3]
    batch_size = 1
    graph_features = []
    
    for batch_idx in range(batch_size):
        edges, edge_types = [], []
        
        # Проходим по всем строкам матрицы, пропуская нулевые ребра
        for i in range(graph_obs.shape[0]):
            source, target, edge_type = graph_obs[i]
            # Пропускаем нулевые ребра (source=0, target=0, edge_type=0)
            if source != 0 or target != 0 or edge_type != 0:
                edges.append((source, target))
                edge_types.append(edge_type)
        
        if not edges:
            graph_embedding = torch.zeros(self.features_dim // 2)
        else:
            # Остальной код обработки графа остается таким же
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            
            unique_nodes = torch.unique(edge_index)
            x = self.node_embedding(unique_nodes)
            
            for layer in self.rgcn_layers:
                if isinstance(layer, nn.Sequential):
                    x = layer[0](x, edge_index, edge_type)
                    x = layer[1](x)
                    x = layer[2](x)
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
        net_arch=[64, 64]  # Упростили архитектуру
    )
    
    model = PPO(
        "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("Начинаем обучение...")
    model.learn(total_timesteps=100000)
    return model

# Тренировка модели
try:
    model = train_ppo(menv)
    model.save("ppo_custom_model")
    
    # Оценка модели
    print("Оценка модели...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        menv, 
        n_eval_episodes=5,
        deterministic=True
    )
    
    print(f"Средняя награда: {mean_reward:.2f} +/- {std_reward:.2f}")
    
except Exception as e:
    print(f"Ошибка при обучении: {e}")
    import traceback
    traceback.print_exc()