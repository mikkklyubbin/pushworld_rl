import torch
import torchrl
# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from pushworld.multiagent_env import MultiAgentPushTargetEnv, Permute
from pushworld.gym_env import PushTargetEnv
from pushworld.solvers import get_check_k_fun, solve_by_model
from torchrl.modules import MultiAgentConvNet
from pushworld.eval import eval_ac
# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
import wandb
# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from pushworld.callbacks import StatsCallback, MetricsCallback
from pushworld.gym_env import INFORMATION_CHANEL_PER_OBJECT, INFORMATION_CHANEL_STATIC
solver = get_check_k_fun(5)
# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm
path_to_rep = "/home/mikk/PushWorld/pushworld_rl/pushworld-main/"
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
print(rgb)
in_channels = 3#0fb1149 e917652
if not rgb:
    in_channels = max_obj * INFORMATION_CHANEL_PER_OBJECT + INFORMATION_CHANEL_STATIC # change 5 to max obj locally
print("in_channels", in_channels)
model_kwargs = {"in_channels": in_channels}

config_train = {"node_feature": 64, "features_dim": 512, "hidden_dim": 512, "batch_size": 128, "n_epochs": 2, "need_pddl":need_pddl,}
print(rgb)
menv = PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100, augment = True, **config)


eval_env =  PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/test", 100, **config)
name_of_test = "test_learning_NOCON_AUGMENT_NEWACT_LOOPPEN"
wandb.init(project="test_", config={**config_train, **config},name="clear_pushworld")
model_save_path = path_to_rep + "python3/model/bst2"

test_ac = []
train_ac = []

def test_model(model):
    test_env = PushTargetEnv(path_to_rep + f"benchmark/puzzles/level0/all/test", 100,  **config)

    num_episodes = 200
    s1 = eval_ac(test_env, num_episodes, model, verbose=True)
    test_ac.append(s1)
    test_env =PushTargetEnv(path_to_rep + "benchmark/puzzles/level0/all/train", 100, seq = True, **config)
    num_episodes = 200
    s1 = eval_ac(test_env, num_episodes, model, verbose=True)
    train_ac.append(s1)
    fig, ax = plt.subplots()
    # plt.figure(figsize=(8, 5))
    ax.plot([i for i in range(len(test_ac))], test_ac)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy')
    fig.savefig(path_to_rep + "python3/fotos/test_ac.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    #plt.figure(figsize=(8, 5))
    ax.plot([i for i in range(len(train_ac))], train_ac)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    fig.savefig(path_to_rep + "python3/fotos/train_ac.png")
    wandb.log({"test_ac": test_ac[-1], "train_ac": train_ac[-1]})
    plt.close(fig)
# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
env_device = "cpu"  # The device where the simulator is run (VMAS can run on GPU)

# Sampling
frames_per_batch = 6_000  # Number of team frames collected per training iteration
n_iters = 5  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 30  # Number of optimization steps per training iteration
minibatch_size = 400  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

# disable log-prob aggregation
set_composite_lp_aggregate(False).set()
max_steps = 100  # Episode steps before done
num_vmas_envs = (
    frames_per_batch // max_steps
)  # Number of vectorized envs. frames_per_batch should be divisible by this number
n_agents = 5

env = MultiAgentPushTargetEnv(menv, solver, device=env_device)

env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)
print(env.rollout(3)[("agents", "observation")].shape)
share_parameters_policy = True

policy_net = torch.nn.Sequential(
    Permute(),
    MultiAgentConvNet(
        in_features=env.observation_spec["agents", "observation"][0][0].shape[-1], 
        n_agents=env.n_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=share_parameters_policy,
        kernel_sizes=3,
        num_cells=[32, 256, 256],
        device=device,
        paddings=[1,1,1],
        activation_class=torch.nn.Tanh,
    ),
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(start_dim=-3),
    torch.nn.Linear(256, env.full_action_spec[env.action_key].space.n)
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "logits")],
)

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec_unbatched,
    in_keys=[("agents", "logits")],
    out_keys=[env.action_key],
    distribution_class=torch.distributions.Categorical,
    return_log_prob=True,
)

share_parameters_critic = True
mappo = True  # IPPO if False

critic_net = torch.nn.Sequential(
    Permute(),
    MultiAgentConvNet(
        in_features=env.observation_spec["agents", "observation"][0][0].shape[-1],
        n_agents=env.n_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=share_parameters_critic,
        kernel_sizes=3,
        num_cells=[32, 256, 256],
        paddings=[1,1,1],
        device=device,
),
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(start_dim=-3),
    torch.nn.Linear(256, 1)                           
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")],
)

collector = SyncDataCollector(
    env,
    policy,
    device=env_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this size
)

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coeff=entropy_eps,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)
loss_module.set_keys(  # We have to tell the loss where to find the keys
    reward=env.reward_key,
    action=env.action_key,
    value=("agents", "state_value"),
    # These last 2 keys will be expanded to match the reward shape
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)


loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)  # We build GAE
GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)

pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

episode_reward_mean_list = []
for tensordict_data in collector:
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )  # Compute GAE and add it to the data

    data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )  # Optional

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()

    # Logging
    done = tensordict_data.get(("next", "agents", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)
    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()