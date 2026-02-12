from pushworld.gym_env import PushTargetEnv

def get_check_k_fun(k:int):
    def check_all_actions(env:PushTargetEnv):
        res = [0 for i in range(k)]
        max_res = (-100, None)
        while (res != [3 for i in range(k)]):
            env.restart()
            rew = 0
            for el in res:
                o,r,te,tr,info = env.step(el)
                rew += r
                if (tr):
                    assert False
                if (te):
                    break
            max_res = max(max_res, (rew, env.current_puzzle.state))
            for i in range(k):
                res[i] = (res[i] + 1) % 4
                if (res[i] != 0):
                    break
        return max_res
    return check_all_actions


def solve_by_model(model, env:PushTargetEnv, k:int):
    env.restart()
    for  i in range(k):
        