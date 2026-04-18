from pushworld.gym_env import PushTargetEnv
from pushworld.gym_env import calc_pathes
import numpy as np
class OnlyPuzDimenssions():
    def __init__(self, a, b) -> None:
        self.dimensions = (a, b)

def check_calc_pathes():
    good = [[1, 0, 1], [1, 1,0], [1, 1, 1]]
    pos = (0, 0)
    f = OnlyPuzDimenssions(3,3)
    ds, par = calc_pathes(good, pos, f)
    assert (ds == np.array([[0, 1e15, 1e15], [1, 2, 1e15], [2, 3, 4]]))
    assert (par[0][0] == -1) and (par[1][0] == (0, 0)) and (par[1][1] == (1, 0))
    assert (par[0][1] == -1) and (par[0][2] == -1) and (par[1][2] == -1)
    assert (par[2][0] == (1, 0))
    