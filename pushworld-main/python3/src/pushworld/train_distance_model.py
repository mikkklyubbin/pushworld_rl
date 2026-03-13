from pushworld.model import ResultPredictor
from torch.utils.data._utils.collate import default_collate
from pushworld.gym_env import PushTargetEnv
import torch 
from torch import nn
import numpy as np
num_epochs = 10
from pushworld.gym_env import INFORMATION_CHANEL_PER_OBJECT, INFORMATION_CHANEL_STATIC
path_to_rep = "/home/mik/hse/Pushworld/pushworld-main/"
use_concentrtion:bool = False
new_actions_rew:float = 0
block_rew:float = 0
block_peny:float = 0
use_block =  False
loop_penalty = 0.05
rgb = False
need_pddl = False
use_MDP = True
use_DIRECT = False
max_obj = 5
print("sss", rgb)

def get_fix_len_roll(menv, episiode):
    obs_data = []
    actions = []
    targets  = []
    terminated = True
    truncated = True
    obs = None
    for i in range(episiode):
        if (terminated or truncated):
            obs, inf = menv.reset()
        true_indices = np.nonzero(obs["av"])
        selected_index = np.random.choice(len(true_indices[0]), size=1)[0]
        selected_index = true_indices[0][selected_index]
        obs_data.append(obs)
        actions.append(selected_index)
        obs, reward, terminated, truncated, info = menv.step(selected_index)
        targets.append(obs["cell"][:, :, 1:2])
    obs_data = default_collate(obs_data)
    actions = torch.tensor(actions)
    targets = torch.tensor(targets)
    return obs_data, actions, targets


config = {
    "use_concentrtion": use_concentrtion,
    "new_actions_rew": new_actions_rew,
    "block_rew": block_rew,
    "block_peny": block_peny, 
    "use_block":use_block,
    "loop_penalty":loop_penalty,
    "need_pddl":need_pddl,
    "to_height":11,
    "to_width":11,
    "max_obj":max_obj,
    "rgb": rgb,
    "use_MDP": use_MDP,
    "use_DIRECT": use_DIRECT,
}
menv = PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100, augment = True, **config)
print(rgb)
in_channels = 3
if not rgb:
    in_channels = menv.all_chanells
print("in_channels", in_channels)

config_train = {"node_feature": 64, "features_dim": 512, "hidden_dim": 512,  "need_pddl":need_pddl,"in_channels": in_channels}
print(menv._observation_space["cell"].shape[0])
model = ResultPredictor(config_train, menv._observation_space, menv._action_space.n, menv._observation_space["cell"].shape[0:3])

num_epochs = 10
for j in  range(num_epochs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    obs_batch, act_batch, targets = get_fix_len_roll(menv, 200)
    for j in range(10):
        optimizer.zero_grad()
        pred = model(obs_batch, act_batch)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        print(loss.item())

save_path = path_to_rep + "python3/model/distance_model"
torch.save(model.enc.state_dict(), save_path)
    