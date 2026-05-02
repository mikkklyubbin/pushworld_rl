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
from gym.wrappers import FrameStack
import numpy as np
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
from pushworld.rendering import savergb
HISTORY_PER_OBJ = 1
INFORMATION_CHANEL_PER_OBJECT = 2 + NUM_ACTIONS + HISTORY_PER_OBJ
HISTORY_STACK = 8

INFORMATION_CHANEL_STATIC = 1 + HISTORY_STACK

def smart_check(good, x, y):
    return (x >= 0 and y >= 0 and x < len(good) and y < len(good[0]) and good[x][y])

def calc_pathes(good, my_pos, puz):
    distance = np.zeros(puz.dimensions) + 1e15
    par = np.zeros((puz.dimensions[0], puz.dimensions[1], 2))-1
    distance[my_pos[0]][my_pos[1]] = 0
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
            if (x + dx >= 0 and x + dx < n and y + dy < m and y + dy >= 0 and distance[x + dx][y + dy] > 1 + distance[x][y] and good[x + dx][y + dy]):
                q.put((x + dx, y + dy))
                distance[x + dx][y + dy] = 1 + distance[x][y]
                par[x + dx][y + dy] = (x, y)
        
    return (distance, par)

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
        max_steps: Optional[int|None] = None,
        border_width: int = DEFAULT_BORDER_WIDTH,
        pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
        standard_padding: bool = False,
        need_pddl:bool = False,
        to_height = None,
        to_width = None,
        max_obj:Optional[int|None] = None,
        seq = False,
        augment = False,
        rgb = True,
        lstm = False,
    ) -> None:
        self._puzzles = []
        self.pddl = need_pddl
        self.puzzle_path = puzzle_path
        self.augment = augment
        self.augment_timer = 200
        self.lstm = lstm
        self.is_obj_pushed = False
        for puzzle_file_path in iter_files_with_extension(
            puzzle_path, PUZZLE_EXTENSION
        ):
            self._puzzles.append(PushWorldPuzzle(puzzle_file_path, self.augment))
        self.curid= 0
        self.seq = seq
        if len(self._puzzles) == 0:
            raise ValueError(f"No PushWorld puzzles found in: {puzzle_path}")
        if border_width < 1:
            raise ValueError("border_width must be >= 1")
        if pixels_per_cell < 3:
            raise ValueError("pixels_per_cell must be >= 3")
        if (max_steps is None):
            max_steps = 400
        self._max_steps = max_steps
        self._pixels_per_cell = pixels_per_cell
        self._border_width = border_width
        widths, heights = zip(*[puzzle.dimensions for puzzle in self._puzzles])
        objs = [len(puzzle._movable_objects) for puzzle in self._puzzles]
        self._max_objs = max(objs)
        self._max_cell_width = max(widths)
        self._max_cell_height = max(heights)
        self.par = None
        self.distance = None
        self.distance_pushable = [None, None, None, None]
        if to_height is not None:
            assert to_height >= self._max_cell_height
            self._max_cell_height=  to_height
        if to_width is not None:
            assert to_width >= self._max_cell_width
            self._max_cell_width=  to_width
        if max_obj is not None:
            assert max_obj >= self._max_objs
            self._max_objs = max_obj
        self.obj_traectory = np.full((self._max_objs, self._max_steps), fill_value=NUM_ACTIONS)
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
        self.pred_static = None
        self.static_distance = None
        self._current_puzzle = None
        self._current_state = None
        self.static_good = None
        self.dynamic_good = None
        self.stack = None
        self.finished = False
        self.rgb = rgb
        self._action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.last_moves = 4 * np.ones((self._max_steps), dtype=np.int32)
        cells_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=render_observation_padded(
                self._puzzles[0], self._puzzles[0].initial_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
            ).shape,
            dtype=np.float32,
        )
        self.all_chanells = 3
        if not self.rgb:
            self.chanels_per_obs = (INFORMATION_CHANEL_STATIC + INFORMATION_CHANEL_PER_OBJECT * self._max_objs)
            self.all_chanells = self.chanels_per_obs
            print("Using non-rgb observation space")
            h,w  = self._max_cell_height + 1, self._max_cell_width + 1
            cells_space = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(w, h, self.all_chanells), dtype=np.float32
            )
            self.agent_observation_space = gym.spaces.Box(
                low=0.0, high=self._max_objs,
                shape=(w, h, self.all_chanells  + INFORMATION_CHANEL_PER_OBJECT + 2), dtype=np.float32
            )
            self.global_observation_space = gym.spaces.Box(
                low=0.0, high=self._max_objs,
                shape=(w, h, self.all_chanells + 1), dtype=np.float32
            )
        self._observation_space = cells_space
        if (self.pddl):
            max_nodes = self._max_cell_height * self._max_cell_width
            max_edges = 11 * max_nodes

            self._observation_space = gym.spaces.Dict({
                'cell': cells_space,
                'edges': gym.spaces.Box(
                    low=0,
                    high=max_nodes-1, 
                    shape=(max_edges, 2),
                    dtype=np.int32
                ), 
                'types':gym.spaces.Box(
                    low=0,
                    high=13, 
                    shape=(max_edges,),
                    dtype=np.int32
                )
            })
        self.obj_pathes = None
        self.obj_blocked = [False for i in range(self._max_objs)]
        

    def set_goals(self, goals):
        self.current_puzzle._goal_state = goals[1:]
        self.current_puzzle.set_agent_goal(goals[0])

    

    def restart(self):
        self._current_state = self._current_puzzle._initial_state
        self.dynamic_good = None
        self.pred_static = None
        self.distance = None
        self.obj_traectory = np.full((self._max_objs, self._max_steps), fill_value=NUM_ACTIONS)
        self.distance_pushable = [None, None, None, None]
        self.par = None
        self.finished = False
        self.prev_av = None
        self.is_obj_pushed = False
        self.last_moves = 4 * np.ones((self.max_steps), dtype=np.int32)
        self.stack = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, self.all_chanells), dtype=np.float32)
        self._steps = 0
        self.obj_pathes = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, self._max_objs))
        self.obj_blocked = [False for i in range(self._max_objs)]
    def collect_multy_agent_data(self, positions, n_agents):
        s = self.get_observation()
        res = np.zeros((n_agents,) + self.agent_observation_space.shape, dtype=np.float32)
        all_show_cur_g  = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, 1), dtype=np.float32)
        for i in range(len(positions)):
            for (x,y) in  self.get_all_cells(self.current_puzzle._movable_objects[i], positions[i]):
                all_show_cur_g[x][y][0] += 1
        for i in range(len(positions)):
            tm = np.concatenate((s, self.get_obj_observation(i)), axis=2)
            tm = np.concatenate((tm, all_show_cur_g), axis=2)
            show_cur_g  = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, 1), dtype=np.float32)
            for (x,y) in  self.get_all_cells(self.current_puzzle._movable_objects[i], positions[i]):
                show_cur_g[x][y][0] += 1
            tm = np.concatenate((tm, show_cur_g), axis=2)
            res[i] = tm
        return res
    
    def get_global_ma_data(self, positions):
        s = self.get_observation()
        all_show_cur_g  = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, 1), dtype=np.float32)
        for i in range(len(positions)):
            for (x,y) in  self.get_all_cells(self.current_puzzle._movable_objects[i], positions[i]):
                all_show_cur_g[x][y][0] += 1
        return np.concatenate((s, all_show_cur_g), axis=2)


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
    
    def check_blocknap(self, dx, dy, obj_idx):
        block = True
        for (x,y) in self.get_all_obj(self.current_puzzle._movable_objects[obj_idx], self._current_state[obj_idx]):
            if (x + dx, y + dy) in self.get_all_obj(self.current_puzzle._movable_objects[obj_idx], self._current_state[obj_idx]) or (x + dx, y + dy) in self.current_puzzle.wall_positions:
                continue
            block = False
            break
        return block
    
    def check_is_blocked(self, obj_idx):
        if (self.obj_blocked[obj_idx]):
            return 0
        cntx = 0
        cnty = 0
        for dx in [-1, 1]:
            if (self.check_blocknap(dx, 0, obj_idx)):
                cntx += 1
        for dy in [-1, 1]:
            if (self.check_blocknap(0, dy, obj_idx)):
                cnty += 1
        if (cntx > 0 and cnty > 0):
            self.obj_blocked[obj_idx] = True
            return -1 * (obj_idx <= len(self.current_puzzle._goals) and self.current_puzzle._goal_state[obj_idx - 1] != self._current_state[obj_idx])
        return 0

    
    def get_relations_graph(self):
        edges = []
        types = []
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
                types.append(bool(flags[x][y] & 4))
                edges.append((self.code_ver(x, y),self.code_ver(x, y)))
                types.append(2 + bool(flags[x][y] & 2))
                edges.append((self.code_ver(x, y),self.code_ver(x, y)))
                types.append(4 + bool(flags[x][y] & 1))

        for x in range(self._max_cell_width):
            for y in range(self._max_cell_height):
                cnt = 0
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    if (self.val_point(x + dx, y + dy) and (flags[x][y] & 4) == 0 and (flags[x + dx][y + dy] & 4) == 0):
                        edges.append((self.code_ver(x, y), self.code_ver(x + dx, y + dy)))
                        types.append(6 + cnt)
                    cnt += 1
        for i in range(0, len(self.current_puzzle.movable_objects)):
            for x, y in self.get_all_obj(self.current_puzzle.movable_objects[i], self._current_state[i]):
                cnt = 0
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    if ((x + dx, y + dy) in  self.get_all_obj(self.current_puzzle.movable_objects[i], self._current_state[i])):
                        edges.append((self.code_ver(x, y), self.code_ver(x + dx, y + dy)))
                        types.append(10 + cnt)
                    cnt += 1
        max_edges = 11 * self._max_cell_height * self._max_cell_width
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
        
    def good_point(self, x:int, y:int, dx:int, dy:int,puz:PushWorldPuzzle):
        if (x + dx  < 0 ):
            return False
        if (y + dy  < 0):
            return False
        if (x + dx >= puz.dimensions[0]):
            return False
        if (y + dy >= puz.dimensions[1]):
            return False
        return True
        
        
    def calc_dists(self, state):
        puz = self.current_puzzle
        que  = queue.Queue()
        ss = np.zeros((puz.dimensions[0], puz.dimensions[1])) + 1e15
        for i in range(len(self.current_puzzle.movable_objects)):
            for el in  self.get_all_cells(self.current_puzzle._movable_objects[i], state[i]):
                ss[el[0]][el[1]] = 0
                que.put((el[0], el[1]))
        while not que.empty():
            f = que.get()
            x, y = f[0], f[1]
            if (x == state[AGENT_IDX][0] and y == state[AGENT_IDX][1]):
                return ss[x][y]
            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                if (self.good_point(x, y, dx, dy, puz) and ss[x + dx][y + dy] > ss[x][y] + 1):
                    ss[x + dx][y + dy] = ss[x][y] + 1
                    que.put((x + dx, y + dy))
        assert False, "Agent is not reachable from any object"
            

    def get_static_block(self):
        if (self.static_good is not None):
            return
        state = self._current_state
        assert state is not None
        my_pos = state[AGENT_IDX]
        puz = self.current_puzzle
        mv_b = self.current_puzzle.movable_objects
        block = np.zeros(puz.dimensions)
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
        self.static_good = good_m
    
    def get_dynamic_good(self):
        if (self.dynamic_good is not None):
            return
        self.get_static_block()
        state = self._current_state
        assert state is not None
        my_pos = state[AGENT_IDX]
        puz = self.current_puzzle
        mv_b = self.current_puzzle.movable_objects
        block = np.zeros(puz.dimensions)
        for i in range(len(mv_b)):
            if (i != AGENT_IDX):
                for el in self.get_all_cells(mv_b[i], state[i]):
                    if (el[0] >= 0 and el[0] < puz.dimensions[0] and el[1] >= 0 and el[1] <puz.dimensions[1]):
                        block[el[0]][el[1]] += 1
        good_m = 1 - np.zeros(puz.dimensions)
        for i in range(0, puz.dimensions[0]):
            for j in range(0, puz.dimensions[1]):
                if not(self.static_good[i][j]):
                    good_m[i][j] = 0
                    continue
                all_cells = subtract_from_points(mv_b[AGENT_IDX].cells, (-i, -j))
                good_m[i][j] = 1
                for x, y in all_cells:
                    if (x < 0 or y < 0 or x >= puz.dimensions[0] or y  >= puz.dimensions[1] or block[x][y]):
                        good_m[i][j] = 0
                        break
        self.dynamic_good = good_m
    def get_static_path(self):
        if (self.pred_static is not None):
            return
        self.get_static_block()
        state = self._current_state
        my_pos = state[AGENT_IDX]
        puz = self.current_puzzle
        good_m = self.static_good
        self.static_distance, self.pred_static = calc_pathes(good_m, my_pos, puz)


        
    def get_all_cells(self, ob:PushWorldObject, pos):
        dx, dy = pos
        return set((x + dx, y + dy) for x, y in ob.cells)
    
    def get_point_to_go(self, id:int, action:int, distances):
        dx, dy = Actions.DISPLACEMENTS[action % 4]
        mv_b = self.current_puzzle.movable_objects
        st = self._current_state
        puz = self.current_puzzle
        all_cells = subtract_from_points(mv_b[AGENT_IDX].cells, ( -dx, -dy))
        an_cells = subtract_from_points(mv_b[id].cells, (-st[id][0], -st[id][1]))
        res = (1e12, -1, -1)
        for el in all_cells:
            for el2 in an_cells:
                #el[0] + i = el2[0] => el2[0] - el[0] = i
                i:int = int(el2[0] - el[0])
                j:int  = int(el2[1] - el[1])
                if (0 <= i and i < puz.dimensions[0] and 0 <= j and j < puz.dimensions[1] and (i,j) not in self.current_puzzle._agent_collision_map[action % 4]):
                    if (distances[i][j] < res[0]):
                        res = (distances[i][j], i, j)
        return res[1:]
    
    def get_interesting_points(self, id:int, action:int, distances):
        self.get_matrix_reachability()
        mv_b = self.current_puzzle.movable_objects
        zz = (distances < 1e12).astype(int)
        zz[mv_b[AGENT_IDX].position[0]][mv_b[AGENT_IDX].position[1]] = 2
        # print(zz.T)
        dx, dy = Actions.DISPLACEMENTS[action % 4]
        st = self._current_state
        puz = self.current_puzzle
        all_cells = subtract_from_points(mv_b[AGENT_IDX].cells, ( -dx, -dy))
        an_cells = subtract_from_points(mv_b[id].cells, (-st[id][0], -st[id][1]))
        res = []
        for el in all_cells:
            for el2 in an_cells:
                #el[0] + i = el2[0] => el2[0] - el[0] = i
                i:int = int(el2[0] - el[0])
                j:int  = int(el2[1] - el[1])
                if (0 <= i and i < puz.dimensions[0] and 0 <= j and j < puz.dimensions[1] and (i,j) not in self.current_puzzle._agent_collision_map[action % 4]):
                    if (distances[i][j] < 1e12):
                        # print(distances[i][j], i, j)
                        res.append((i, j))
        return res
    
    def find_perimater_map(self, distances):
        mv_b = self.current_puzzle.movable_objects
        perimeter = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1), dtype=np.bool)
        for i in range(1, len(mv_b)):
            for el in self.get_all_cells(self.current_puzzle._movable_objects[i], self._current_state[i]):
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if (el[0] + dx >= 0 and el[0] + dx < self._max_cell_width and el[1] + dy >= 0 and el[1] + dy < self._max_cell_height):
                            perimeter[el[0] + dx][el[1] + dy] = True
        for el in self.current_puzzle.wall_positions:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if (self.val_point(el[0] + dx, el[1] + dy)):
                        perimeter[el[0] + dx][el[1] + dy] = True
        return perimeter


    def get_matrix_reachability(self, verbose = False) -> None:
        if (self.par is not None):
            return
        state = self._current_state
        assert state is not None
        if (verbose):
            print(self._current_state, "curst")
        my_pos = state[AGENT_IDX]
        puz = self.current_puzzle
        self.get_dynamic_good()
        good_m = self.dynamic_good
        self.distance, self.par = calc_pathes(good_m, my_pos, puz)
        
    def get_obj_observation(self, id:int) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_puzzle is not None
        self.get_matrix_reachability()
        res = np.zeros((self._max_cell_width + 1,self._max_cell_height + 1, INFORMATION_CHANEL_PER_OBJECT), dtype=np.float32)
        for el in  self.get_all_cells(self.current_puzzle._movable_objects[id], self._current_state[id]):
            res[el[0]][el[1]][0] = 1.0
        assert self.current_puzzle._goal_state 
        if (id -1 < len(self.current_puzzle._goals) and id > 0):
            for el in self.get_all_cells(self.current_puzzle._goals[id - 1], self._current_puzzle._goal_state[id - 1]):
                res[el[0]][el[1]][1] = 1.0
        self.get_matrix_reachability()
        for  i in range(0, 4):
            self.calc_cells_for_push(i)
            x, y = self.get_point_to_go(id, i, self.distance_pushable[i])
            back = self.par
            if (x, y) == (-1, -1):
                self.get_static_path()
                x,y = self.get_point_to_go(id, i, self.static_distance)
                back = self.pred_static
            while (x != -1):
                x, y = int(x), int(y)
                res[x][y][2 + i] = 1.0
                x, y =  back[x][y] 
        res[:, :, 2 + NUM_ACTIONS] = self.obj_pathes[:,:, id]
        return res
        
        
    def get_observation(self):
        assert self._current_state is not None 
        assert self._current_puzzle is not None 
        observation = render_observation_padded(
            self._current_puzzle, self._current_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
        )
        if (not self.rgb):
            observation = np.zeros((self._max_cell_width + 1,self._max_cell_height + 1, INFORMATION_CHANEL_STATIC), dtype=np.float32)
            for i in range(len(self.current_puzzle._movable_objects)):
                for el in self.get_all_cells(self.current_puzzle._movable_objects[i], self._current_state[i]):
                    observation[el[0]][el[1]][0] = 1.0 
                    observation[el[0]][el[1]][1] = 1.0 
                for el in  self._current_puzzle._wall_positions:
                    observation[el[0]][el[1]][0] = 1.0
                observation = np.concatenate((observation, self.get_obj_observation(i)), axis=2)
            observation = np.concatenate((observation, np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, INFORMATION_CHANEL_PER_OBJECT * (self._max_objs - len(self.current_puzzle._movable_objects))), dtype=np.float32)), axis=2)
            observation[:,:, INFORMATION_CHANEL_STATIC - HISTORY_STACK + 1: INFORMATION_CHANEL_STATIC] = self.stack[:,:,:-1]
            self.stack = observation[:,:, INFORMATION_CHANEL_STATIC - HISTORY_STACK : INFORMATION_CHANEL_STATIC]
        if (self.pddl):
            gr  =  self.get_relations_graph()
            return {
                'cell': observation,
                'edges': gr["edges"],
                'types': gr['types']
            }

        return observation


    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Union[np.ndarray, Dict[str, Any]], Dict[str, Any]]:
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
        self.augment_timer-=1
        if (self.augment == 1 and self.augment_timer == 0):
            self.augment_timer = 200
            self._puzzles.clear()
            for puzzle_file_path in iter_files_with_extension(
                self.puzzle_path, PUZZLE_EXTENSION
            ):
                self._puzzles.append(PushWorldPuzzle(puzzle_file_path, self.augment))
        self.par = None
        self.distance = None
        self.distance_pushable = [None, None, None, None]
        self.static_good = None
        self.dynamic_good = None
        self.pred_static = None
        self.static_distance = None
        self.finished = False
        self.is_obj_pushed = False
        self.obj_traectory = np.full((self._max_objs, self._max_steps), fill_value=NUM_ACTIONS)
        self.last_moves =4 * np.ones((self.max_steps), dtype=np.int32)
        self.obj_blocked = [False for i in range(self._max_objs)]
        self._current_puzzle = self._random_generator.choice(self._puzzles)
        self.obj_pathes = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, self._max_objs))
        if self.seq:
            # print(self.curid)
            self._current_puzzle = self._puzzles[self.curid % len(self._puzzles)]
            self.curid += 1
        self._current_state = self._current_puzzle.initial_state
        self._current_achieved_goals = self._current_puzzle.count_achieved_goals(
            self._current_state
        )
        self._steps = 0
        self.stack = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, HISTORY_STACK), dtype=np.float32)
        return self.get_obs_and_info()
    
    def get_obs_and_info(self):
        observation = render_observation_padded(
            self._current_puzzle, self._current_state, self._max_cell_height, self._max_cell_width, self._pixels_per_cell, self._border_width,
        )
        info = {"puzzle_state": self._current_state}
        observation = self.get_observation()
        return observation, info
    
    def change_state(self, new_state):
        self._current_state = new_state
        self.obj_is_moved()
        self.agent_is_moved()
        self.stack = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, HISTORY_STACK), dtype=np.float32)
        self._steps = 0
        self.obj_pathes = np.zeros((self._max_cell_width + 1, self._max_cell_height + 1, self._max_objs))
        self.finished = False
        self.obj_blocked = [False for i in range(self._max_objs)]
    
    def obj_is_moved(self):
        self.prev_av = None
        self.distance = None
        self.distance_pushable = [None, None, None, None]
        self.par = None
        self.dynamic_good = None
    
    def agent_is_moved(self):
        self.obj_traectory = np.full((self._max_objs, self._max_steps), fill_value=NUM_ACTIONS)
        self.prev_av = None
        self.distance = None
        self.distance_pushable = [None, None, None, None]
        self.par = None
        self.static_distance = None
        self.pred_static = None


    def step(
        self,
        action: int,
        fast: bool = False,
    ) -> Union[
        Tuple[Optional[Union[np.ndarray, Dict[str, Any]]], float, bool, dict],
        Tuple[Optional[Union[np.ndarray, Dict[str, Any]]], float, bool, bool, dict],
    ]:
        """Implements `gym.Env.step`.

        The returned observation is an RGB image of the new state of the environment,
        formatted as a `float32` array with shape (height, width, 3) and values ranging
        from [0, 1].
        """
        assert self._current_puzzle is not None, "reset() must be called before step() can be called."
        assert self._current_state is not None, "reset() must be called before step() can be called."
        if not self._action_space.contains(action):
            raise ValueError("The provided action is not in the action space.")

        if self._current_state is None:
            raise RuntimeError("reset() must be called before step() can be called.")
        if (self._steps < len(self.last_moves)):
            self.last_moves[self._steps] = action
        self._steps += 1
        previous_state = self._current_state
        self._current_state, count = self._current_puzzle.get_next_state(
            self._current_state, action
        )
        for i in range(len(self.current_puzzle.movables)):
            if (self.current_puzzle.was_moved[i]):
                self.obj_traectory[i][self._steps -1] = action

        self.is_obj_pushed = (count > 1)
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
            prev_dists = self.calc_dists(previous_state) * 0.01
            cur_dists = self.calc_dists(self._current_state) *  0.01
            
            cur_distance = self._current_puzzle.count_sum_distance(
                self._current_state
            )
            current_achieved_goals = self._current_puzzle.count_achieved_goals(
                self._current_state
            )

            reward = current_achieved_goals - previous_achieved_goals - 0.01 + previous_distance - cur_distance -cur_dists + prev_dists
            for i in range(1, len(self.current_puzzle._movable_objects)):
                reward += self.check_is_blocked(i)

        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        info = {"puzzle_state": self._current_state}
        if (self.is_obj_pushed):
            self.obj_is_moved()
        if (count != 0):
            self.agent_is_moved()
        for i in range(len(self._current_state)):
            x,y = self._current_state[i]
            self.obj_pathes[x][y][i] = 1
        if (fast and not terminated and not truncated):
            return None, reward, terminated, truncated, info
        observation = self.get_observation()
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
        obs = self.get_observation()
        terminated = self._current_puzzle.is_goal_state(self._current_state)
        reward = -0.01
        if terminated:
            reward = 10.0
        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        info = {"puzzle_state": self._current_state}
        return obs, reward, terminated, truncated, info


    def pushing_graph(self, action:int):
        self.get_matrix_reachability()
        self.used = [False for i in range(self._max_objs)]
        self.pushable = [True for i in range(self._max_objs)]
        for i in range(1, len(self.current_puzzle._movable_objects)):
            if (not self.used[i]):
                self.dfs(i, action)
    
    def calc_cells_for_push(self, action:int):
        if (self.distance_pushable[action % 4] is not None):
            return
        self.pushing_graph(action)
        dx = Actions.DISPLACEMENTS[action % 4][0]
        dy = Actions.DISPLACEMENTS[action % 4][1]
        self.distance_pushable[action % 4] = self.distance.copy()
        moved = self.get_all_obj(self.current_puzzle._movable_objects[0], (dx,dy))
        for i in range(1, len(self.current_puzzle._movable_objects)):
            if (not self.pushable[i]):
                for el in self.get_all_obj(self.current_puzzle._movable_objects[i], self._current_state[i]):
                    for el2 in moved:
                        pos = (el[0] - el2[0], el[1] - el2[1])
                        if (0 <= pos[0] and pos[0] < self.distance_pushable[action % 4].shape[0] and 0 <= pos[1] and pos[1] < self.distance_pushable[action % 4].shape[1]):
                            self.distance_pushable[action % 4][pos[0]][pos[1]] = 1e12




    def get_collision(self, id_move, dx, dy, obj_idx, move_pos):
        obj_moved = self.get_all_obj(self.current_puzzle._movable_objects[id_move], (move_pos[0] + dx, move_pos[1] + dy))   
        for (x,y) in self.get_all_obj(self.current_puzzle._movable_objects[obj_idx], self._current_state[obj_idx]):
            if (x, y) in obj_moved:
                return True
        return False

    def dfs(self, id:int, action:int):
        self.used[id] = True
        dx = Actions.DISPLACEMENTS[action % 4][0]
        dy = Actions.DISPLACEMENTS[action % 4][1]
        if (self._current_state[id] in self.current_puzzle._wall_collision_map[action][id]):
            self.pushable[id] = False
            return
        for i in range(1, len(self.current_puzzle._movable_objects)):
            if (self.get_collision(id, dx, dy, i, self._current_state[id])):
                if  not self.used[i]:
                    self.dfs(i, action)
                self.pushable[id] = self.pushable[id] and self.pushable[i]
                if (not self.pushable[id]):
                    return

        
        

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
        augment = False,
        loop_penalty =0,
        new_actions_rew = 0,
        use_concentrtion = True,
        use_block = False,
        block_rew = 0, 
        block_peny = 0,
        need_pddl = False,
        rgb = True,
        use_MDP = True,
        use_DIRECT = False,
        lstm = False,
        max_env_steps = 30, 
        teleport = False,
        sub_models_list = [],
        length_list = []
    ) -> None:
        super().__init__(puzzle_path, max_steps, border_width, pixels_per_cell, standard_padding, to_height=to_height, to_width=to_width, seq=seq, augment=augment, need_pddl=need_pddl, rgb=rgb, max_obj=max_obj, lstm=lstm)
        self.max_mov_ob = 0
        self.use_concentrtion = use_concentrtion
        self.max_steps = max_steps
        self.new_actions_rew = new_actions_rew
        self.teleport = teleport
        self.sub_model_list = sub_models_list
        self.length_list = length_list
        self.acts = []
        self.use_block = use_block
        self.hash_history = {}
        self.env_steps = 0
        self.loop_penalty = loop_penalty
        self.use_DIRECT = use_DIRECT
        self.block_rew = block_rew
        self.block_peny = block_peny
        self.block = None
        self.max_env_steps = max_env_steps
        self.last_ac_is_direct = False
        self.use_MDP = use_MDP
        if (self.use_block):
            assert(self.block_peny  >= 0)
        assert(self.loop_penalty >= 0)
        for el in self._puzzles:
            self.max_mov_ob = max(self.max_mov_ob, len(el._movable_objects))
        if (max_obj is not None):
            assert max_obj >= self.max_mov_ob
            self.max_mov_ob = max_obj
        self.dir_shift = self.max_mov_ob * NUM_ACTIONS + self.max_mov_ob * NUM_AD_ACTIONS
        self.env_acts = self.max_mov_ob * NUM_ACTIONS + self.max_mov_ob * NUM_AD_ACTIONS + use_DIRECT * (self._max_cell_height + 1) * (self._max_cell_width + 1)
        self._action_space = gym.spaces.Discrete(self.max_mov_ob * NUM_ACTIONS + self.max_mov_ob * NUM_AD_ACTIONS + use_DIRECT * (self._max_cell_height + 1) * (self._max_cell_width + 1) + len(self.sub_model_list))
        boss = super().observation_space
        mat1_ob = boss["cell"] if self.pddl else boss
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
        last_ac = gym.spaces.Box(
            low=0.0,
            high=NUM_ACTIONS + 1,
            shape=(self._max_steps, ),
            dtype=np.int32,
        )
        av = gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=bool)
        self._observation_space = gym.spaces.Dict({
            'cell': mat1_ob,
            'positions': pos_ob,
            'av': av,
            'last_ac': last_ac
        })
        if (self.pddl):
            self._observation_space = gym.spaces.Dict({
                'cell': mat1_ob,
                'positions': pos_ob,
                'av': av,
                'last_ac': last_ac,
                'edges':boss["edges"], 
                'types':boss["types"], 
            })
        self.prev_av = None
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
    
    def sub_modal_act(self, action):
        action -= self.env_acts
        assert action < len(self.sub_model_list), "No such action"
        obs, info = self.get_obs_and_info()
        obs = self.convert(obs)
        reward = 0
        for i in range(self.length_list[action]):
            step_action = self.sub_model_list[action]
            obs,  rew, terminated, truncated, info = self.step(step_action)
            reward += rew
            truncated, terminated, info = self.trans_term_truncated(truncated, terminated, info, obs)
            if (terminated or truncated):
                return obs, reward, terminated, truncated, info
        return obs, reward, False, False, info 
    
    def move_to_point(self,x, y, need_last = False):
        self.get_matrix_reachability()
        rew = 0
        act = self.get_action_list(x, y)
        self.add = tuple(self._current_state)
        # print(act)
        self.acts += act
        penalty = 0
        cnt = 0
        for el in act:
            tmp = tuple(self._current_state)
            penalty += self.rec_alst_moves(el) * self.loop_penalty * int(not(self.is_obj_pushed))
            observation, reward, terminated, truncated, info = super().step(el, fast=(cnt < len(act) - 1) or (not need_last))
            if terminated or truncated:
                info["terminal_observation"] = self.convert(observation)
            else:
                info["terminal_observation"] = None
            if (truncated):
                return self.convert(observation), rew + reward - penalty, terminated, truncated, info
            if (tmp[1:] != self._current_state[1:]):
                assert(False)
            if (terminated):
                #actions should be design, like not finish actions
                raise LookupError
            rew += reward
            cnt += 1

        return self.convert(self.get_observation()), rew - penalty, False  , False, {}

    def check_not_stupid_point(self, x, y):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for i in range(1, len(self.current_puzzle._movable_objects)):
                    for el in self.get_all_obj(self.current_puzzle._movable_objects[i], self._current_state[i]):
                        if (el[0] == x + dx and el[1] == y + dy):
                            return True
                for el in self.current_puzzle._wall_positions:
                    if (el[0] == x + dx and el[1] == y + dy):
                        return True
                return False
            
    
    
    
    def get_av_act(self):
        if (self.prev_av is not None):
            return self.prev_av

        av = np.zeros(self.action_space.n, dtype=bool)
        mv_b = self.current_puzzle.movable_objects
        av[self.max_mov_ob * NUM_ACTIONS:self.max_mov_ob * NUM_ACTIONS + len(mv_b) * 2] = self.use_concentrtion
        av[self.max_mov_ob * NUM_ACTIONS + self.max_mov_ob * 2 + 1: self.max_mov_ob * NUM_ACTIONS + self.max_mov_ob * 2 + len(mv_b)] = self.use_block
        for  a in  range(4):
            self.calc_cells_for_push(a)
        if (self.use_DIRECT and not self.last_ac_is_direct):
            shift = self.max_mov_ob * NUM_ACTIONS + self.max_mov_ob * NUM_AD_ACTIONS
            self.get_matrix_reachability()
            f = self.get_all_interesting(self.distance).reshape(-1)
            av[shift: shift + f.shape[0]] = f
        st = self._current_state
        per = None
        if (self.teleport):
            self.get_matrix_reachability()
            per = self.find_perimater_map(self.distance)
        for a in range(4):
            av[a] = ((st[0][0], st[0][1]) not in self.current_puzzle._agent_collision_map[a])
            dx, dy = Actions.DISPLACEMENTS[a]
            
            if (self.teleport and av[a]):
                good = False
                for el in  self.get_all_cells(mv_b[AGENT_IDX], (dx + st[0][0], dy + st[0][1])):
                    if (smart_check(per, el[0], el[1])):
                        good = True
                        break
                if (not good): 
                    av[a] = False
            if (self.use_DIRECT and not self.last_ac_is_direct):
                av[a] = False
        self.get_matrix_reachability()
        assert self.distance is not None
        puz = self.current_puzzle
        for action in range(4, len(mv_b) * 4):
            # print(action // 4)
            # print(self.current_puzzle._block[action // 4])
            if ((st[action // 4][0], st[action // 4][1]) in self.current_puzzle._wall_collision_map[action % 4][action // 4]) or self.current_puzzle._block[action // 4] or not(self.use_MDP): #or self.current_puzzle._block[action // 4]
                continue
            dx, dy = Actions.DISPLACEMENTS[action % 4]
            good = False
            all_cells = subtract_from_points(mv_b[AGENT_IDX].cells, ( -dx, -dy))
            an_cells = subtract_from_points(mv_b[action // 4].cells, (-st[action // 4][0], -st[action // 4][1]))

            for el in all_cells:
                for el2 in an_cells:
                    #el[0] + i = el2[0] => el2[0] - el[0] = i
                    i:int = int(el2[0] - el[0])
                    j:int  = int(el2[1] - el[1])
                    if (0 <= i and i < puz.dimensions[0] and 0 <= j and j < puz.dimensions[1] and (i,j) not in self.current_puzzle._agent_collision_map[action % 4]):
                        if (self.distance_pushable[action % 4][i][j] < 1e12):
                            good = True
                            break
                if (good):
                    break
            av[action] = good
            
        assert(av in self.observation_space["av"])

        if (av.sum() == 0):
            print(av)
            rgb = self.render()
            savergb(rgb, "/home/mik/hse/Pushworld/pushworld-main/python3/1.jpg")
            print("no av")
            self.get_av_act()
        self.prev_av = av
        assert av.sum() > 0, "No available actions"

        return av


    def get_current_pos(self):
        assert self._current_puzzle is not None
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
    ) -> Tuple[Union[np.ndarray, Dict[str, Any]], Dict[str, Any]]:
        """Implements `gym.Env.reset`.

        This function randomly selects a puzzle from those provided to the constructor
        and resets the environment to the initial state of the puzzle.
observation
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
        self.prev_av = None
        self.hash_history = {}
        self.block = None
        self.last_ac_is_direct = False
        # for i in range(mat1['cell'].shape[2]):
        #     print(mat1['cell'][:,:,i])
        # print(mat1["av"])
        obs = self.convert(mat1)
        self.env_steps = 0
        info["terminal_observation"] = None
        if (self.convert(mat1)["last_ac"] not in self.observation_space["last_ac"]):
            print(self.convert(mat1)["last_ac"].shape)
            print(self.observation_space["last_ac"])
            assert(self.convert(mat1)["last_ac"] in self.observation_space["last_ac"])
        assert(self.convert(mat1) in self.observation_space)
        assert(obs in self.observation_space)
        return obs, info
    def convert(self, observation):
        if (self.pddl):
            return {
                'cell': observation['cell'],
                'edges':observation['edges'],
                'types':observation['types'],
                'positions': self.get_current_pos(),
                'av':self.get_av_act(),
                'last_ac':self.last_moves,
            }

        return {
            'cell': observation,
            'positions': self.get_current_pos(),
            'av':self.get_av_act(),
            'last_ac':self.last_moves,
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
    

    def rec_alst_moves(self, action:int):
        action %= 4
        if (self.last_moves[self._steps - 1] // 2 == action // 2 and self.last_moves[self._steps - 1] != action):
            return 1
        return 0
    
    def gen_empty_action_res(self):
        observation, info  = self.get_obs_and_info()
        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        info = {}
        if truncated:
            info["terminal_observation"] = self.convert(observation)
        else:
            info["terminal_observation"] = None
        # assert("ZZZZ" == "VVV")
        # assert(self.convert(observation) in self.observation_space)
        return self.convert(observation), -1, False, truncated, info
    
    def trans_term_truncated(self, truncated, terminated, info, obs):
        if (terminated and self.lstm):
            self.finished = True
            terminated = False
        truncated = (truncated or (self.env_steps >= self.max_env_steps and (not self.finished)))
        terminated = (terminated or (self.env_steps >= self.max_env_steps and self.finished))
        if terminated or truncated:
            info["terminal_observation"] = self.convert(obs)
        else:
            info["terminal_observation"] = None
        return truncated, terminated, info
        

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
        self.env_steps += 1
        if self.finished:
            observation, _, terminated, truncated, info  = self.get_all_info()
            truncated, terminated, info = self.trans_term_truncated(truncated, terminated, info, observation)
            return self.convert(observation), 0, terminated, truncated, info
        av_delta = -self.get_av_act().sum()
        self.prev_av = None
        mv_b = self.current_puzzle.movable_objects
        prev_st = self._current_state
        if (action >= self.dir_shift):
            action  -= self.dir_shift

            x = (action) // (self._max_cell_width + 1)
            y = (action) % (self._max_cell_width + 1)
            self.last_ac_is_direct = True

            tmp = self.move_to_point(x, y)

            return tmp
        self.last_ac_is_direct = False

        if (action >= NUM_ACTIONS * self.max_mov_ob):
            self._steps += 1
            action = action - NUM_ACTIONS * self.max_mov_ob
            self.acts.append(action + NUM_ACTIONS)
            dlt = 0
            if (action % self.max_mov_ob >= len(mv_b)):
                return self.gen_empty_action_res()
            if (action >= 2 * self.max_mov_ob):
                if (not self.use_block):
                    return self.gen_empty_action_res()
                self.current_puzzle.change_block(action - 2 * self.max_mov_ob)
                self.prev_av = None
            elif (action % 2 == 1):
                if (not self.use_concentrtion):
                    return self.gen_empty_action_res()
                self.current_puzzle.concentrate(action // 2)
            else:
                if (not self.use_concentrtion):
                    return self.gen_empty_action_res()
                self.current_puzzle.deconcentrate(action // 2)
            observation, reward, terminated, truncated, info = self.get_all_info()
            truncated, terminated, info = self.trans_term_truncated(truncated, terminated, info, observation)
            return self.convert(observation), -0.01, terminated, truncated, info
        if (action // 4 == 0):
            self.acts.append(action)
            penalty = self.rec_alst_moves(action) * self.loop_penalty * int(not(self.is_obj_pushed))
            observation, reward, terminated, truncated, info = super().step(action % 4)
            self.prev_av = None
            obs = self.convert(observation)
            av_delta += obs["av"].sum()
            for i in range(len(prev_st)):
                if (self.current_puzzle._block[i] and (self._current_state[i] != prev_st[i])):
                    reward -= self.block_peny
            reward += self.new_actions_rew * av_delta
            reward -= penalty
            reward += sum(self.current_puzzle._block) * self.block_rew
            truncated, terminated, info = self.trans_term_truncated(truncated, terminated, info, observation)
            return obs, reward, terminated, truncated, info
        optimal = (1e15, -1, -1)
        self.get_matrix_reachability()
        rew = 0
        if (action // 4 < len(mv_b)):
            self.calc_cells_for_push(action % 4)
            optimal = self.get_point_to_go(action // 4, action % 4, self.distance_pushable[action % 4])
            if (optimal != (-1, -1)):
                obs, rew, terminated, truncated, info = self.move_to_point(optimal[0], optimal[1], need_last=self.teleport)
                if (truncated or terminated):
                    return obs, rew, terminated, truncated, info
                if (not  self.teleport):
                    self.acts.append(action % 4)
                    rew -= self.rec_alst_moves(action % 4) * self.loop_penalty* int(not(self.is_obj_pushed))
                    obs , reward, terminated, truncated, info = super().step(action % 4)
                    obs = self.convert(obs)
                    rew += reward

                self.prev_av = None
                av_delta += obs["av"].sum()
                rew += self.new_actions_rew * av_delta
                for i in range(len(prev_st)):
                    if (self.current_puzzle._block[i] and (self._current_state[i] != prev_st[i])):
                        rew -= self.block_peny
                rew += self.new_actions_rew * av_delta
                rew += sum(self.current_puzzle._block) * self.block_rew
                truncated, terminated, info = self.trans_term_truncated(truncated, terminated, info, obs)
                return obs, rew, terminated, truncated, info

            else:
                print("warning: no path")
                rew = -1
        elif (action < self.env_acts + len(self.sub_model_list)):
            return self.sub_modal_act(action)
        else:
            print("warning: ??")
            rew = -1
        return self.gen_empty_action_res()
    
    def get_all_interesting(self,distances):
        res = np.zeros((self._max_cell_height + 1, self._max_cell_width + 1), dtype=bool)
        #print(res.shape)
        for i in range(NUM_ACTIONS):
            for id in range (1, len(self.current_puzzle._movable_objects)):
                c = self.get_interesting_points(id, i, distances)
                for el in c:
                    # print(el, i, id)
                    res[el[0]][el[1]] = True
        return res
                

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
    
    def render_acts(
        self,
        border_width: int = DEFAULT_BORDER_WIDTH,
        pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
    ):
        """Creates a video of the given plan, starting from the initial state.
        Args:
            plan: A sequence of actions.
            border_width: The pixel width of the border drawn to indicate object
                boundaries. Must be >= 1.
            pixels_per_cell: The pixel width and height of a discrete position in the
                environment. Must be >= 1 + 2 * border_width.
        Returns:
            A list of RGB images with shape (height, width, 3) and type `uint8`.
        """

        self.current_puzzle.state = self.current_puzzle._initial_state
        self.current_puzzle._colors = [1.0 for i in range(len(self.current_puzzle.movable_objects))]
        state = self.current_puzzle._initial_state
        image = self.current_puzzle.render(
            state=state,
            border_width=border_width,
            pixels_per_cell=pixels_per_cell,
        )
        images = [image]
        for action in self.acts:
            if (action >= NUM_ACTIONS):
                action -= NUM_ACTIONS
                if (action >= 2 * self.max_mov_ob):
                    self.current_puzzle.change_block(action - 2 * self.max_mov_ob)
                elif (action % 2 == 1):
                    self.current_puzzle.concentrate(action // 2)
                else:
                    self.current_puzzle.deconcentrate(action // 2)
            else:
                self.current_puzzle.state, _ = self.current_puzzle.get_next_state(self.current_puzzle.state, action)
            image = self.current_puzzle.render(
                state=self.current_puzzle.state,
                border_width=border_width,
                pixels_per_cell=pixels_per_cell,
            )
            images.append(image)
        return images

    def render_video(self):
        return self.current_puzzle.render_plan(self.acts)