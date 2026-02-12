
from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import (
    BoundedContinuous,
    Composite,
    Categorical,
    UnboundedContinuous,
)
import torch
from torchrl.envs.utils import check_env_specs
from pushworld.gym_env import PushTargetEnv
class MultiAgentPushTargetEnv(EnvBase):
    def __init__(self, env : PushTargetEnv, solver, device="cpu", batch_size=[]):
        super().__init__(device=device, batch_size=batch_size)
        self.env = env
        self.solver = solver
        self.n_agents = self.env.max_mov_ob
        self.max_steps = 10
        self._make_specs()

    def _make_specs(self):
        self.observation_spec = Composite({
            "agents": Composite({
                "observation":BoundedContinuous(
                shape=(self.n_agents, *self.env.agent_observation_space.shape),
                dtype=torch.float32,
                low=0,
                high=self.n_agents,
            )}, shape=(self.n_agents,)),
            "global":BoundedContinuous(
                shape=self.env.global_observation_space.shape,
                dtype=torch.float32,
                low=0,
                high=self.n_agents,
            )
        }, shape=())
        self.action_spec = Composite({
            "agents": Composite({
                "action": Categorical(4, shape=(self.n_agents,))
            }, shape=(self.n_agents,))
        }, shape=())
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": UnboundedContinuous(shape=(self.n_agents,))
            }, shape=(self.n_agents,))
        }, shape=())
        self.done_spec = Composite({
            "done": Categorical(1, dtype=torch.bool, shape=(self.n_agents, 1)),
            "terminated": Categorical(1, dtype=torch.bool, shape=(self.n_agents, 1)),
            "truncated": Categorical(1, dtype=torch.bool, shape=(self.n_agents, 1)),
        }, shape=(),
            device=self.device,)
        self.positions = None

    def calc_cur_f(self):
        self.env.restart()
        tmp = self.env.current_puzzle._goal_state
        tmp = (None, *tmp)
        self.env.set_goals(self.positions)
        r1, state = self.solver(self.env)
        self.env.change_state(state)
        self.env.set_goals(tmp)
        r2, state = self.solver(self.env)
        return r1 + r2
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)


    def _reset(self, tensordict=None):
        obs, _ = self.env.reset()
        self.positions = self.env._current_state
        self.positions = [list(t) for t in self.positions]
        self.step_count = 0
        self.cur  = self.calc_cur_f()
        agents_data = self.env.collect_multy_agent_data(self.positions, self.n_agents)
        global_data = self.env.get_global_ma_data(self.positions)

        self.borders = []
        for i in range(self.n_agents):
            self.borders.append([100000, -100000, 100000, -100000])
            for el in self.env.current_puzzle._movable_objects[i].cells:
                self.borders[-1][0] = min(self.borders[-1][0], el[0])
                self.borders[-1][1] = max(self.borders[-1][1], el[0])
                self.borders[-1][2] = min(self.borders[-1][2], el[1])
                self.borders[-1][3] = max(self.borders[-1][3], el[1])
        agents_data  = TensorDict({
            "observation": self.env.collect_multy_agent_data(self.positions, self.n_agents),
        }, batch_size=[self.n_agents], device=self.device)
        out = TensorDict({
            "agents": agents_data,
            "global": global_data,
        }, batch_size=self.batch_size,
            device=self.device,)
        return out
    
    def _step(self, tensordict):
        actions = tensordict[("agents", "action")]
        reward = torch.zeros(self.n_agents, 1)
        for i, act in enumerate(actions):
            if (i >= len(self.env.current_puzzle._movable_objects)):
                break
            if act == 0 and self.positions[i][1] + self.borders[i][3] + 1 <= self.env._max_cell_height - 1:
                self.positions[i][1] = min(self.positions[i][1] + 1, self.env._max_cell_height   - 1)
            elif act == 1 and self.positions[i][1] + self.borders[i][2] - 1 >= 0:
                self.positions[i][1] = max(self.positions[i][1] - 1, 0)
            elif act == 2 and self.positions[i][0] + self.borders[i][0] - 1 >= 0:
                self.positions[i][0] = max(self.positions[i][0] - 1, 0)
            elif act == 3 and self.positions[i][0] + self.borders[i][1] + 1 <= self.env._max_cell_width - 1:
                self.positions[i][0] = min(self.positions[i][0] + 1, self.env._max_cell_width - 1)
            dlt = self.calc_cur_f() - self.cur
            self.cur += dlt
            reward[i, 0] = dlt
        self.step_count += 1
        truncated = torch.full((self.n_agents, 1), self.step_count >= self.max_steps, dtype=torch.bool)
        terminated = torch.full((self.n_agents, 1), False, dtype=torch.bool)
        done = terminated | truncated 
        agents_data  = TensorDict({
            "observation": self.env.collect_multy_agent_data(self.positions, self.n_agents),
            "reward": reward,
        }, batch_size=[self.n_agents], device=self.device)


        out = TensorDict({
            "agents": agents_data,
            "global": self.env.get_global_ma_data(self.positions),
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }, batch_size=self.batch_size,
            device=self.device,)
        return out
    
def stupid_solver(env):
    return 0, env._current_state


tf = MultiAgentPushTargetEnv(PushTargetEnv("/home/mik/hse/Pushworld/pushworld-main/benchmark/puzzles/level0/all/test/level_0_all_test_199.pwp", 100, rgb=False, use_concentrtion=False, use_MDP=False), stupid_solver)

print(tf.action_keys)
print(tf.reward_keys)
print(tf.observation_keys)
print(tf.done_keys)
check_env_specs(tf)