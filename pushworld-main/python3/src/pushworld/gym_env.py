# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import cv2
import queue
from pushworld.config import PUZZLE_EXTENSION
from pushworld.puzzle import (
    DEFAULT_BORDER_WIDTH,
    DEFAULT_PIXELS_PER_CELL,
    NUM_ACTIONS,
    NUM_AD_ACTIONS,
    AGENT_IDX,
    PushWorldPuzzle,
    PushWorldObject,
    subtract_from_points,
    Actions
)
from pushworld.utils.env_utils import get_max_puzzle_dimensions, render_observation_padded
from pushworld.utils.filesystem import iter_files_with_extension


class PushWorldEnv(gym.Env):
    """An OpenAI Gym environment for PushWorld puzzles.

    Rewards are calculated according to Appendix D of
    https://arxiv.org/pdf/1707.06203.pdf with one change: The negative reward per step
    is reduced to 0.01, since PushWorld puzzles tend to have longer solutions than
    Sokoban puzzles.

    Args:
        puzzle_path: The path of a PushWorld puzzle file or of a directory that
            contains puzzle files, possibly nested in subdirectories. All discovered
            puzzles are loaded, and the `reset` method randomly selects a new puzzle
            each time it is called.
        max_steps: If not None, the `step` method will return `done = True` after
            calling it `max_steps` times since the most recent call of `reset`.
        border_width: The pixel width of the border drawn to indicate object
            boundaries. Must be >= 1.
        pixels_per_cell: The pixel width and height of a discrete position in the
            environment. Must be >= 1 + 2 * border_width.
        standard_padding: If True, all puzzles are padded to the maximum width and
            height of the puzzles in the `pushworld.config.BENCHMARK_PUZZLES_PATH`
            directory. If False, puzzles are padded to the maximum dimensions of
            all puzzles found in the `puzzle_path`.
    """

    def __init__(
        self,
        puzzle_path: str,
        max_steps: Optional[int] = None,
        border_width: int = DEFAULT_BORDER_WIDTH,
        pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
        standard_padding: bool = False,
        need_pddl:bool = False,
        to_height = None,
        to_width = None,
        seq = False,
    ) -> None:
        self._puzzles = []
        self.pddl = need_pddl
        for puzzle_file_path in iter_files_with_extension(
            puzzle_path, PUZZLE_EXTENSION
        ):
            self._puzzles.append(PushWorldPuzzle(puzzle_file_path))
        self.curid= 0
        self.seq = seq
        if len(self._puzzles) == 0:
            raise ValueError(f"No PushWorld puzzles found in: {puzzle_path}")
        if border_width < 1:
            raise ValueError("border_width must be >= 1")
        if pixels_per_cell < 3:
            raise ValueError("pixels_per_cell must be >= 3")

        self._max_steps = max_steps
        self._pixels_per_cell = pixels_per_cell
        self._border_width = border_width

        widths, heights = zip(*[puzzle.dimensions for puzzle in self._puzzles])
        self._max_cell_width = max(widths)
        self._max_cell_height = max(heights)
        if to_height is not None:
            assert to_height >= self._max_cell_height
            self._max_cell_height=  to_height
        if to_width is not None:
            assert to_width >= self._max_cell_width
            self._max_cell_width=  to_width
        if standard_padding:
            standard_cell_height, standard_cell_width = get_max_puzzle_dimensions()

            if standard_cell_height < self._max_cell_height:
                raise ValueError(
                    "`standard_padding` is True, but the maximum puzzle height in "
                    "BENCHMARK_PUZZLES_PATH is less than the height of the puzzle(s) "
                    "in the given `puzzle_path`."
                )
            else:
                self._max_cell_height = standard_cell_height

            if standard_cell_width < self._max_cell_width:
                raise ValueError(
                    "`standard_padding` is True, but the maximum puzzle width in "
                    "BENCHMARK_PUZZLES_PATH is less than the width of the puzzle(s) "
                    "in the given `puzzle_path`."
                )
            else:
                self._max_cell_width = standard_cell_width

        # Use a fixed arbitrary seed for reproducibility of results and for
        # deterministic tests.
        self._random_generator = random.Random(123)

        self._current_puzzle = None
        self._current_state = None

        self._action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self._observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=render_observation_padded(
                self._puzzles[0], self._puzzles[0].initial_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
            ).shape,
            dtype=np.float32,
        )
        if (self.pddl):
            cells_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=render_observation_padded(
                    self._puzzles[0], self._puzzles[0].initial_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
                ).shape,
                dtype=np.float32,
            )
            max_nodes = self._max_cell_height * self._max_cell_width
            max_edges = 8 * max_nodes
            graph_space = gym.spaces.Dict({
                            'edges':gym.spaces.Box(
                low=0,
                high=max_nodes-1, 
                shape=(max_edges, 2),
                dtype=np.int32
            ),
                            'types':gym.spaces.Box(
                low=0,
                high=max_nodes-1, 
                shape=(max_edges),
                dtype=np.int32
            )
                            })
            graph_space = gym.spaces.Box(
                low=0,
                high=max_nodes-1, 
                shape=(max_edges, 3),
                dtype=np.int32
            )

            self._observation_space = gym.spaces.Dict({
                'cell': cells_space,
                'graph': graph_space
            })

    @property
    def action_space(self) -> gym.spaces.Space:
        """Implements `gym.Env.action_space`."""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        """Implements `gym.Env.observation_space`."""
        return self._observation_space

    @property
    def metadata(self) -> Dict[str, Any]:
        """Implements `gym.Env.metadata`."""
        return {"render_modes": ["rgb_array"]}

    @property
    def render_mode(self) -> str:
        """Implements `gym.Env.render_mode`. Always contains "rgb_array"."""
        return "rgb_array"

    @property
    def current_puzzle(self) -> PushWorldPuzzle or None:
        """The current puzzle, or `None` if `reset` has not yet been called."""
        return self._current_puzzle
    
    def val_point(self, x:int, y:int):
        return (x >= 0 and y >= 0 and x < self._max_cell_width and y < self._max_cell_height)
    
    def code_ver(self, x:int, y:int):
        return x * self._max_cell_height + y
    
    def get_all_obj(self, ob:PushWorldObject, pos):
        dx, dy = pos
        return set((x + dx, y + dy) for x, y in ob.cells)
    
    def get_relations_graph(self):
        edges = []
        types = []
        for x in range(self._max_cell_width):
            for y in range(self._max_cell_height):
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    if (self.val_point(x + dx, y + dy)):
                        edges.append((self.code_ver(x, y), self.code_ver(x + dx, y + dy)))
                        types.append(0)
                        for dx2, dy2 in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                            if (self.val_point(x + dx + dx2, y + dy + dy2)):
                                edges.append((self.code_ver(x, y), self.code_ver(x + dx + dx2, y + dy + dy2)))
                                types.append(1)
        flags = [[0 for y in range(self._max_cell_height)]for x in range(self._max_cell_width)]
        for i in range(1, len(self.current_puzzle.movable_objects)):
            for el in self.get_all_obj(self.current_puzzle.movable_objects[i], self._current_state[i]):
                flags[el[0]][el[1]] += 1
        for i in range(0, 1):
            for el in self.get_all_obj(self.current_puzzle.movable_objects[i], self._current_state[i]):
                flags[el[0]][el[1]] += 2
        for el in self.current_puzzle.wall_positions:
            if (el[0] >= 0 and el[0] < self._max_cell_width and el[1] >= 0 and el[1] < self._max_cell_height):
                flags[el[0]][el[1]] += 4
        for x in range(self._max_cell_width):
            for y in range(self._max_cell_height):
                edges.append((self.code_ver(x, y),self.code_ver(x, y)))
                types.append(2 + bool(flags[x][y] & 4))
                edges.append((self.code_ver(x, y),self.code_ver(x, y)))
                types.append(4 + bool(flags[x][y] & 2))
                edges.append((self.code_ver(x, y),self.code_ver(x, y)))
                types.append(6 + bool(flags[x][y] & 1))
        max_edges = 8 * self._max_cell_height * self._max_cell_width
        graph_matrix = np.zeros((max_edges, 2), dtype=np.int32)
        types_res = np.zeros((max_edges), dtype=np.int32)
        num_edges = min(len(edges), max_edges)
        for i in range(num_edges):
            source, target = edges[i]
            graph_matrix[i] = [source, target]
            types_res[i] = types[i]

        return {
                'edges':graph_matrix,
                'types':types_res
        }


    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Implements `gym.Env.reset`.

        This function randomly selects a puzzle from those provided to the constructor
        and resets the environment to the initial state of the puzzle.

        Args:
            seed: If not None, the random number generator in this environment is reset
                with this seed.
            options: Unused. Required by the `gym.Env.reset` interface.

        Returns:
            A tuple of (observation, info). The observation contains the initial
            observation of the environment after the reset, and it is formatted as an
            RGB image with shape (height, width, 3) with `float32` type and values
            ranging from [0, 1]. The info dictionary is unused.
        """
        if seed is not None:
            self._random_generator = random.Random(seed)

        self._current_puzzle = self._random_generator.choice(self._puzzles)
        if self.seq:
            self._current_puzzle = self._puzzles[self.curid % len(self._puzzles)]
            self.curid += 1
        self._current_state = self._current_puzzle.initial_state
        self._current_achieved_goals = self._current_puzzle.count_achieved_goals(
            self._current_state
        )
        self._steps = 0

        observation = render_observation_padded(
            self._current_puzzle, self._current_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
        )
        info = {"puzzle_state": self._current_state}

        if (self.pddl):
            return {
                'cell': observation,
                'graph': self.get_relations_graph()
            }, info

        return observation, info

    def step(self, action: int) -> Union[Tuple[np.ndarray, float, bool, dict], Tuple[np.ndarray, float, bool, bool, dict]]:
        """Implements `gym.Env.step`.

        The returned observation is an RGB image of the new state of the environment,
        formatted as a `float32` array with shape (height, width, 3) and values ranging
        from [0, 1].
        """
        if not self._action_space.contains(action):
            raise ValueError("The provided action is not in the action space.")

        if self._current_state is None:
            raise RuntimeError("reset() must be called before step() can be called.")

        self._steps += 1
        previous_state = self._current_state
        self._current_state = self._current_puzzle.get_next_state(
            self._current_state, action
        )
        observation = render_observation_padded(
            self._current_puzzle, self._current_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
        )

        terminated = self._current_puzzle.is_goal_state(self._current_state)

        if terminated:
            reward = 10.0
        else:
            previous_achieved_goals = self._current_puzzle.count_achieved_goals(
                previous_state
            )
            previous_distance = self._current_puzzle.count_sum_distance(
                previous_state
            )
            cur_distance = self._current_puzzle.count_sum_distance(
                self._current_state
            )
            current_achieved_goals = self._current_puzzle.count_achieved_goals(
                self._current_state
            )
            reward = current_achieved_goals - previous_achieved_goals - 0.01 + previous_distance - cur_distance

        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        info = {"puzzle_state": self._current_state}
        if (self.pddl):
            return {
                'cell': observation,
                'graph': self.get_relations_graph()
            }, reward, terminated, truncated, info

        return observation, reward, terminated, truncated, info

    def render(self, mode='rgb_array') -> np.ndarray:
        """Implements `gym.Env.render`.

        Returns:
            An RGB image of the current state of the environment, formatted as a
            `uint8` array with shape (height, width, 3).
        """
        assert mode == 'rgb_array', 'mode must be rgb_array.'
        return self._current_puzzle.render(
            self._current_state,
            border_width=self._border_width,
            pixels_per_cell=self._pixels_per_cell,
        )
    
    def get_all_info(self):
        obs = render_observation_padded(self._current_puzzle, self._current_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width)
        terminated = self._current_puzzle.is_goal_state(self._current_state)

        if terminated:
            reward = 10.0
        else:
            reward = -0.01

        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        info = {"puzzle_state": self._current_state}
        if (self.pddl):
            return {
                'cell': obs,
                'graph': self.get_relations_graph()
            }, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info
        


def savergb(rgb_array, name):
    if rgb_array.dtype == np.float32 or rgb_array.dtype == np.float64:
        rgb_array = (rgb_array * 255).astype(np.uint8)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, bgr_array)
class PushTargetEnv(PushWorldEnv):
    def __init__(
        self,
        puzzle_path: str,
        max_steps: Optional[int] = None,
        border_width: int = DEFAULT_BORDER_WIDTH,
        pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
        standard_padding: bool = False,
        to_height = None,
        to_width = None,
        max_obj = None,
        seq = False,
    ) -> None:
        super().__init__(puzzle_path, max_steps, border_width, pixels_per_cell, standard_padding, to_height=to_height, to_width=to_width, seq=seq)
        self.max_mov_ob = 0
        self.max_steps = max_steps
        self.acts = []
        for el in self._puzzles:
            self.max_mov_ob = max(self.max_mov_ob, len(el._movable_objects))
        if (max_obj is not None):
            assert max_obj >= self.max_mov_ob
            self.max_mov_ob = max_obj
        self._action_space = gym.spaces.Discrete(self.max_mov_ob * NUM_ACTIONS + self.max_mov_ob * NUM_AD_ACTIONS)
        mat1_ob = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=render_observation_padded(
                self._puzzles[0], self._puzzles[0].initial_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
            ).shape,
            dtype=np.float32,
        )
        #print(self._max_cell_height)
        # print(render_observation_padded(
        #         self._puzzles[0], self._puzzles[0].initial_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
        #     ).shape)
        pos_ob = gym.spaces.Box(
            low=-1.0,
            high=max(self._max_cell_height, self._max_cell_width),
            shape=(self.max_mov_ob, 2),
            dtype=np.float32,
        )
        av = gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=bool)
        self._observation_space = gym.spaces.Dict({
            'cell': mat1_ob,
            'positions': pos_ob,
            'av': av
        })
        #print(self._observation_space['cell'])

    @property
    def action_space(self) -> gym.spaces.Space:
        """Implements `gym.Env.action_space`."""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        """Implements `gym.Env.observation_space`."""
        return self._observation_space

    @property
    def metadata(self) -> Dict[str, Any]:
        """Implements `gym.Env.metadata`."""
        return {"render_modes": ["rgb_array"]}

    @property
    def render_mode(self) -> str:
        """Implements `gym.Env.render_mode`. Always contains "rgb_array"."""
        return "rgb_array"

    @property
    def current_puzzle(self) -> PushWorldPuzzle or None:
        """The current puzzle, or `None` if `reset` has not yet been called."""
        return self._current_puzzle
    
    
    def get_av_act(self):
        av = np.zeros(self.action_space.n, dtype=bool)
        mv_b = self.current_puzzle.movable_objects
        av[self.max_mov_ob * NUM_ACTIONS:self.max_mov_ob * NUM_ACTIONS + len(mv_b) * NUM_AD_ACTIONS] = 1
        av[0] = av[1] = av[2] = av[3] = 1
        st = self._current_state
        self.get_matrix_reachability()
        puz = self.current_puzzle
        for action in range(4, len(mv_b) * 4):
            dx, dy = Actions.DISPLACEMENTS[action % 4]
            good = False
            for i in range(puz.dimensions[0]):
                for j in range(puz.dimensions[1]):
                    all_cells = subtract_from_points(mv_b[AGENT_IDX].cells, (-i -dx, -j -dy))
                    an_cells = subtract_from_points(mv_b[action // 4].cells, (-st[action // 4][0], -st[action // 4][1]))
                    good:bool = False
                    if (self.distance[i][j] < 1e12):
                        for el in all_cells:
                            for el2 in an_cells:
                                if (int(el[0]) == int(el2[0]) and int(el[1]) == int(el2[1])):
                                    good = True
                                    break
                            if (good):
                                break
                        if (good):
                            break
                if (good):
                    break
            av[action] = good
            
        assert(av in self.observation_space["av"])
        return av


    def get_current_pos(self):
        pos = np.full((self.max_mov_ob, 2), -1, dtype=np.float32)
        id:int = 0
        for el in self._current_puzzle._movable_objects:
            x,y = el.position
            pos[id][0] = x
            pos[id][1] = y
            id += 1
        return pos

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Implements `gym.Env.reset`.

        This function randomly selects a puzzle from those provided to the constructor
        and resets the environment to the initial state of the puzzle.

        Args:
            seed: If not None, the random number generator in this environment is reset
                with this seed.
            options: Unused. Required by the `gym.Env.reset` interface.

        Returns:
            A tuple of (observation, info). The observation contains the initial
            observation of the environment after the reset, and it is formatted as an
            RGB image with shape (height, width, 3) with `float32` type and values
            ranging from [0, 1]. The info dictionary is unused.
        """
        mat1, info = super().reset(seed, options)
        self._steps = 0
        self.acts = []
        obs = {
            'cell': mat1,
            'positions': self.get_current_pos(),
            'av': self.get_av_act()
        }
        # print(self.convert(mat1)['cell'].shape)
        # print(self.get_current_pos().shape)
        # print(self.observation_space['positions'])
        # print(self.get_current_pos() in self.observation_space['positions'])
        info["terminal_observation"] = None
        assert(self.convert(mat1) in self.observation_space)
        assert(obs in self.observation_space)
        return obs, info
    
    def get_all_cells(self, ob:PushWorldObject, pos):
        dx, dy = pos
        return set((x + dx, y + dy) for x, y in ob.cells)
    
    def get_matrix_reachability(self, verbose = False):
        state = self._current_state
        if (verbose):
            print(self._current_state)
        my_pos = state[AGENT_IDX]
        puz = self.current_puzzle
        mv_b = self.current_puzzle.movable_objects
        block = np.zeros(puz.dimensions)
        for i in range(len(mv_b)):
            if (i != AGENT_IDX):
                for el in self.get_all_cells(mv_b[i], state[i]):
                    if (el[0] >= 0 and el[0] < puz.dimensions[0] and el[1] >= 0 and el[1] <puz.dimensions[1]):
                        block[el[0]][el[1]] += 1
        for el in puz.wall_positions:
            if (el[0] >= 0 and el[0] < puz.dimensions[0] and el[1] >= 0 and el[1] < puz.dimensions[1]):
                block[el[0]][el[1]] += 1
        for el in puz.agent_wall_positions:
            if (el[0] >= 0 and el[0] < puz.dimensions[0] and el[1] >= 0 and el[1] < puz.dimensions[1]):
                block[el[0]][el[1]] += 1
        good_m = 1 - np.zeros(puz.dimensions)
        self.block=block
        for i in range(0, puz.dimensions[0]):
            for j in range(0, puz.dimensions[1]):
                all_cells = subtract_from_points(mv_b[AGENT_IDX].cells, (-i, -j))
                good_m[i][j] = 1
                for x, y in all_cells:
                    if (x < 0 or y < 0 or x >= puz.dimensions[0] or y  >= puz.dimensions[1] or block[x][y]):
                        good_m[i][j] = 0
                        break
        distance = np.zeros(puz.dimensions) + 1e15
        par = np.zeros((puz.dimensions[0], puz.dimensions[1], 2))-1
        distance[my_pos[0]][my_pos[1]] = 0
        # print(distance[my_pos[0]][my_pos[1]])
        # print(my_pos[0], my_pos[1])
        # print(distance)
        q = queue.Queue()
        q.put(my_pos)
        n = puz.dimensions[0]
        m = puz.dimensions[1]
        while not q.empty():
            f = q.get()
            x,y = f
            for ch in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dx = ch[0]
                dy = ch[1]
                if (x + dx >= 0 and x + dx < n and y + dy < m and y + dy >= 0 and distance[x + dx][y + dy] > 1 + distance[x][y] and good_m[x + dx][y + dy]):
                    if (verbose):
                        print((x + dx, y + dy), (x, y))
                    q.put((x + dx, y + dy))
                    distance[x + dx][y + dy] = 1 + distance[x][y]
                    par[x + dx][y + dy] = (x, y)
                    if (verbose):
                        print(par[x + dx][y + dy])
            for i in range(n):
                for j in range(m):
                    if int(distance[i, j]) < 1e9 and (i, j) != (int(my_pos[0]), int(my_pos[1])):
                        assert par[i, j, 0] != -1 and par[i, j, 1] != -1, f"Cell ({i}, {j}) is reachable but has no parent. Distance: {distance[i, j]}"
            
        self.distance = distance
        self.par = par
        for i in range(n):
            for j in range(m):
                if int(distance[i, j]) < 1e9 and (i, j) != (int(my_pos[0]), int(my_pos[1])):
                    assert par[i, j, 0] != -1 and par[i, j, 1] != -1, f"Cell ({i}, {j}) is reachable but has no parent. Distance: {distance[i, j]}"
                    pass

    def convert(self, observation):
        return {
            'cell': observation,
            'positions': self.get_current_pos(),
            'av':self.get_av_act()
        }

    def get_action_list(self, x:int, y:int):
        act = []
        while (int(self.par[x][y][0]) != -1):
            x1, y1 = self.par[x][y]
            if (x1 < x):
                act.append(1)
            elif x1 > x:
                act.append(0)
            elif y1 > y:
                act.append(2)
            else:
                act.append(3)
            x, y = int(x1),int(y1)
        act.reverse()
        return act


    def step(self, action: int) -> Union[Tuple[np.ndarray, float, bool, dict], Tuple[np.ndarray, float, bool, bool, dict]]:
        """Implements `gym.Env.step`.

        The returned observation is an RGB image of the new state of the environment,
        formatted as a `float32` array with shape (height, width, 3) and values ranging
        from [0, 1].
        """
        if not self._action_space.contains(action):
            raise ValueError("The provided action is not in the action space.")

        if self._current_state is None:
            raise RuntimeError("reset() must be called before step() can be called.")

        self._steps += 1
        if (action >= NUM_ACTIONS * self.max_mov_ob):
            action = action - NUM_ACTIONS * self.max_mov_ob
            if (action % 2 == 1):
                self.current_puzzle.concentrate(action // 2)
            else:
                self.current_puzzle.deconcentrate(action // 2)
            observation, reward, terminated, truncated, info = self.get_all_info()
            if terminated or truncated:
                info["terminal_observation"] = self.convert(observation)
            else:
                info["terminal_observation"] = None
            return self.convert(observation), reward, terminated, truncated, info
            
        if (action // 4 == 0):
            self.acts.append(action)
            observation, reward, terminated, truncated, info = super().step(action % 4)
            if terminated or truncated:
                info["terminal_observation"] = self.convert(observation)
            else:
                info["terminal_observation"] = None
            return self.convert(observation), reward, terminated, truncated, info
        dx, dy = Actions.DISPLACEMENTS[action % 4]
        mv_b = self.current_puzzle.movable_objects
        st = self._current_state
        optimal = (1e15, -1, -1)
        self.get_matrix_reachability()
        puz = self.current_puzzle
        rew = 0
        if (action // 4 < len(mv_b)):
            for i in range(puz.dimensions[0]):
                for j in range(puz.dimensions[1]):
                    all_cells = subtract_from_points(mv_b[AGENT_IDX].cells, (-i -dx, -j -dy))
                    an_cells = subtract_from_points(mv_b[action // 4].cells, (-st[action // 4][0], -st[action // 4][1]))
                    good:bool = False
                    for el in all_cells:
                        for el2 in an_cells:
                            if (int(el[0]) == int(el2[0]) and int(el[1]) == int(el2[1])):
                                good = True
                                break
                    if (good):
                        optimal = min(optimal, (self.distance[i][j], i, j))
            if (int(optimal[0]) < 1e12):
                act = self.get_action_list(optimal[1], optimal[2])
                self.add = tuple(self._current_state)
                # print(act)
                self.acts += act
                for el in act:
                    tmp = tuple(self._current_state)
                    observation, reward, terminated, truncated, info = super().step(el)
                    if terminated or truncated:
                        info["terminal_observation"] = self.convert(observation)
                    else:
                        info["terminal_observation"] = None
                    if (truncated):
                        return self.convert(observation), rew + reward, terminated, truncated, info
                    if (tmp[1:] != self._current_state[1:]):
                        print(act)
                        print(self.block.T)
                        print((self.distance == 1e15).astype(np.float32).T)
                        print(optimal[1])
                        print(optimal[2])
                        self.block[int(optimal[1])][int(optimal[2])] = 9
                        print(self.block.T)
                        x, y = optimal[1:]
                        while (int(self.par[x][y][0]) != -1):
                            print(x, y)
                            x1, y1 = self.par[x][y]
                            if (x1 < x):
                                act.append(1)
                            elif x1 > x:
                                act.append(0)
                            elif y1 > y:
                                act.append(3)
                            else:
                                act.append(2)
                            x, y = int(x1),int(y1)
                            print(self.par[x][y])
                            print((x,y))
                        self.get_matrix_reachability(verbose=True)
                        savergb(self.render(), "2.jpg")
                        self._current_state = tmp
                        savergb(self.render(), "1.jpg")
                        assert(False)
                    if (terminated):
                        raise LookupError
                        print("XXX")
                    rew += reward
                self.acts.append(action % 4)
                observation, reward, terminated, truncated, info = super().step(action % 4)
                if terminated or truncated:
                    info["terminal_observation"] = self.convert(observation)
                else:
                    info["terminal_observation"] = None
                assert(self.convert(observation) in self.observation_space)
                return self.convert(observation), reward + rew, terminated, truncated, info
            else:
                rew = -1
        else:
            rew = -1
        observation = render_observation_padded(
            self.current_puzzle, self._current_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
        )
        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        info = {}
        if truncated:
            info["terminal_observation"] = self.convert(observation)
        else:
            info["terminal_observation"] = None
        assert(self.convert(observation) in self.observation_space)
        return self.convert(observation), rew, False, truncated, info

    def render(self, mode='rgb_array') -> np.ndarray:
        """Implements `gym.Env.render`.

        Returns:
            An RGB image of the current state of the environment, formatted as a
            `uint8` array with shape (height, width, 3).
        """
        assert mode == 'rgb_array', 'mode must be rgb_array.'
        return self._current_puzzle.render(
            self._current_state,
            border_width=self._border_width,
            pixels_per_cell=self._pixels_per_cell,
        )
    def render_video(self):
        return self.current_puzzle.render_plan(self.acts)