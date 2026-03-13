import numpy as np
def eval_ac(env, num_episodes:int, model, verbose:bool = False):
    model.policy.set_training_mode(False)
    success_count:int = 0
    for episode in range(num_episodes):
        obs, _ = env.reset()  
        terminated = False
        truncated = False
        episode_rewards = []
        while not terminated:
            action, _ = model.predict(obs)  

            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            if (truncated):
                break
        if terminated:
            success_count += 1
    if verbose:
        print(f"\nРезультаты за {num_episodes} эпизодов:")
        print(f"Успешных эпизодов: {success_count}")
        print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
    s1 = success_count/num_episodes*100
    model.policy.set_training_mode(True)
    return s1



def eval_ac_rec(env, num_episodes:int, model, verbose:bool = False):
    model.policy.set_training_mode(False)
    success_count:int = 0
    for episode in range(num_episodes):
        obs, _ = env.reset()  
        terminated = False
        truncated = False
        episode_rewards = []
        lstm_states = None
        episode_starts = True
        while not terminated:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=False
            )
            episode_starts = False
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            if (truncated):
                break
        if terminated:
            success_count += 1
    if verbose:
        print(f"\nРезультаты за {num_episodes} эпизодов:")
        print(f"Успешных эпизодов: {success_count}")
        print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
    s1 = success_count/num_episodes*100
    model.policy.set_training_mode(True)
    return s1