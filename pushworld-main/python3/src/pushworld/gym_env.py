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
import queue
from pushworld.config import PUZZLE_EXTENSION
from pushworld.puzzle import (
    DEFAULT_BORDER_WIDTH,
    DEFAULT_PIXELS_PER_CELL,
    NUM_ACTIONS,
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
    ) -> None:
        self._puzzles = []
        for puzzle_file_path in iter_files_with_extension(
            puzzle_path, PUZZLE_EXTENSION
        ):
            self._puzzles.append(PushWorldPuzzle(puzzle_file_path))

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
        self._current_state = self._current_puzzle.initial_state
        self._current_achieved_goals = self._current_puzzle.count_achieved_goals(
            self._current_state
        )
        self._steps = 0

        observation = render_observation_padded(
            self._current_puzzle, self._current_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
        )
        info = {"puzzle_state": self._current_state}

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
            current_achieved_goals = self._current_puzzle.count_achieved_goals(
                self._current_state
            )
            reward = current_achieved_goals - previous_achieved_goals - 0.01

        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        info = {"puzzle_state": self._current_state}

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



class PushTargetEnv(PushWorldEnv):
    def __init__(
        self,
        puzzle_path: str,
        max_steps: Optional[int] = None,
        border_width: int = DEFAULT_BORDER_WIDTH,
        pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
        standard_padding: bool = False,
    ) -> None:
        super().__init__(puzzle_path, max_steps, border_width, pixels_per_cell, standard_padding)
        self._random_generator = super()._random_generator
        self.max_mov_ob = 0
        self.max_steps = max_steps
        for el in super()._puzzles:
            self.max_mov_ob = max(self.max_mov_ob, len(el._movable_objects))
        
        self._action_space = gym.spaces.Discrete(self.max_mov_ob * NUM_ACTIONS)
        mat1_ob = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=render_observation_padded(
                self._puzzles[0], self._puzzles[0].initial_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
            ).shape,
            dtype=np.float32,
        )
        pos_ob = gym.spaces.Box(
            low=-1.0,
            high=max(self._max_cell_height, self._max_cell_width),
            shape=(self.max_mov_ob, 2),
            dtype=np.float32,
        )
        self._observation_space = gym.spaces.Dict({
            'cell': mat1_ob,
            'positions': pos_ob
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
        return super()._current_puzzle


    def get_current_pos(self):
        pos = np.full((self.max_mov_ob, 2), -1)
        id:int = 0
        for el in super()._current_puzzle._movable_objects:
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

        obs = {
            'cell': mat1,
            'positions': self.get_current_pos()
        }
        return obs, info
    
    def get_all_cells(self, ob:PushWorldObject, pos):
        dx, dy = ob.position
        return set((x - dx, y - dy) for x, y in ob.cells)
    
    def get_matrix_reachability(self):
        state = super()._current_state
        my_pos = state[AGENT_IDX]
        puz = self.current_puzzle
        mv_b = self.current_puzzle.movable_objects
        block = np.zeros(puz.shape)
        for i in range(len(mv_b)):
            if (i != AGENT_IDX):
                for el in self.get_all_cells(mv_b[i], state[i]):
                    block[el[0]][el[1]] += 1
        for el in puz.wall_positions:
            block[el[0]][el[1]] += 1
        for el in puz.agent_wall_positions:
            block[el[0]][el[1]] += 1
        good_m = 1 - np.zeros(puz.shape)
        for i in range(0, puz.shape[0]):
            for j in range(0, puz.shape[1]):
                all_cells = subtract_from_points(mv_b[AGENT_IDX], (-i, -j))
                good_m[i][j] = 1
                for x, y in all_cells:
                    if (block[x][y]):
                        good_m[i][j] = 0
                        break
        distance = np.zeros(puz.shape) + 1e15
        par = np.zeros((puz.shape[0], puz.shape[1], 2))-1
        distance[my_pos[0]][my_pos[1]] = 0
        q = queue.Queue()
        n = puz.shape[0]
        m = puz.shape[1]
        while q.empty():
            f = q.get()
            x,y = f
            for ch in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dx = ch[0]
                dy = ch[1]
                if (x + dx >= 0 and x + dx < n and y + dy < m and y - dy >= 0 and distance[x + dx][y + dy] > 1 + distance[x][y] and good_m[x + dx][y + dy]):
                    q.put((x + dx, y + dy))
                    distance[x + dx][y + dy] = 1 + distance[x][y]
                    par[x + dx][y + dy] = (x, y)
        self.distance = distance
        self.par = par
        



    def get_action_list(self, x, y):
        act = []
        while (self.par[x][y] != (-1, -1)):
            x1, y1 = self.par[x][y]
            if (x1 < x):
                act.append(1)
            elif x1 > x:
                act.append(0)
            elif y1 > y:
                act.append(2)
            else:
                act.append(3)
            x, y = x1,y1
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

        if super()._current_state is None:
            raise RuntimeError("reset() must be called before step() can be called.")

        self._steps += 1
        if (action // 4 == 0):
            return super().step(action % 4)
        dx, dy = Actions.DISPLACEMENTS[action % 4]
        mv_b = self.current_puzzle.movable_objects
        st = super()._current_state
        optimal = (1e15, -1, -1)
        self.get_matrix_reachability()
        puz = self.current_puzzle
        rew = 0
        if (action // 4 < len(mv_b)):
            for i in range(puz.shape[0]):
                for j in range(puz.shape[1]):
                    all_cells = subtract_from_points(mv_b[AGENT_IDX], (-i -dx, -j -dy))
                    an_cells = subtract_from_points(mv_b[action // 4].cells, st[action // 4])
                    good:bool = False
                    for el in all_cells:
                        for el2 in an_cells:
                            if (el == el2):
                                good = True
                                break
                    if (good):
                        optimal = max(optimal, (self.distance[i][j], i, j))
            if (optimal[0] != 1e15):
                act = self.get_action_list(optimal[1], optimal[2])
                for el in act:
                    observation, reward, terminated, truncated, info = super().step(el)
                    if (truncated):
                        return observation, rew + reward, terminated, truncated, info
                    if (terminated):
                        raise LookupError
                    rew += reward
                observation, reward, terminated, truncated, info = super().step(action % 4)
                return observation, reward + rew, terminated, truncated, info
            else:
                rew = -1
        else:
            rew = -1
        observation = render_observation_padded(
            self.current_puzzle, super()._current_state, super()._max_cell_height, super()._max_cell_width, super()._pixels_per_cell, super()._border_width,
        )
        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        return observation, rew, False, truncated

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