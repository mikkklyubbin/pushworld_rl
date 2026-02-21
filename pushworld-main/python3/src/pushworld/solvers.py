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
            max_res = max(max_res, (rew, env._current_state))
            for i in range(k):
                res[i] = (res[i] + 1) % 4
                if (res[i] != 0):
                    break
        return max_res
    return check_all_actions
def get_solver_by_model(model, k:int):
    def solve_by_model(env:PushTargetEnv):
        env.restart()
        o, i = env.get_obs_and_info()
        o = env.convert(o)
        rw = 0
        for j in range(k):
            action, _ = model.predict(o)
            o, r, te, tr, info = env.step(action)
            if (te or tr):
                break
            rw += r
        return rw, env._current_state
    return solve_by_model