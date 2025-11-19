
from pushworld.gym_env import PushWorldEnv
path_to_rep = "/home/mikk/PushWorld/pushworld_rl/pushworld-main/"
test_env = PushWorldEnv(path_to_rep + f"benchmark/puzzles/level0/all/test", 100)

num_episodes = 200
success_count = 0
for episode in range(num_episodes):
    obs, _ = test_env.reset()  
    terminated = False
    truncated = False
    episode_rewards = []
    while not terminated:
        action = test_env.action_space.sample()
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_rewards.append(reward)
        if (truncated):
            break
    if terminated:
        print(1)
        print(episode)
        rgb = test_env.render()
        success_count += 1
print(f"\nРезультаты за {num_episodes} эпизодов:")
print(f"Успешных эпизодов: {success_count}")
print(f"Процент успеха: {success_count/num_episodes*100:.2f}%")
s1 = success_count/num_episodes*100
