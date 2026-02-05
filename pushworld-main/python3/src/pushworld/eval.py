def eval_ac(env, num_episodes:int, model, verbose:bool = False):
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
    return s1