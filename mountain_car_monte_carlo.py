import random
from collections import defaultdict
import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

grid_size = 5
position_grid = np.linspace(-1.2, 0.6, grid_size)
velocity_grid = np.linspace(-0.07, 0.07, grid_size)


def discretize_state(state):
    position_index = np.digitize(state[0], position_grid) - 1
    velocity_index = np.digitize(state[1], velocity_grid) - 1
    position_index = max(0, min(position_index, grid_size - 1))
    velocity_index = max(0, min(velocity_index, grid_size - 1))
    return position_index, velocity_index

def epsilon_greedy_policy(state, Q, num_actions, epsilon=0.5):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        return max(range(num_actions), key=lambda a: Q[(state, a)])


def generate_episode(env, Q, num_timesteps=1000, epsilon=0.5):
    episode = []
    state, _ = env.reset()

    for t in range(num_timesteps):
        discrete_state = discretize_state(state)

        action = epsilon_greedy_policy(discrete_state, Q, env.action_space.n, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        reward = -1 / (np.abs(next_state[0]) * np.abs(next_state[1]) + 1)

        episode.append((discrete_state, action, reward))

        if terminated or truncated:
            break

        state = next_state

    return episode

def monte_carlo(env, num_iterations=10000, epsilon=0.7, epsilon_decay=0.999):
    Q = defaultdict(float)
    total_return = defaultdict(float)
    N = defaultdict(int)

    for i in range(num_iterations):
        episode = generate_episode(env, Q, num_timesteps=10000, epsilon=epsilon)
        epsilon *= epsilon_decay

        all_state_action_pairs = [(s, a) for (s, a, r) in episode]
        rewards = [r for (s, a, r) in episode]

        for t, (state, action, reward) in enumerate(episode):
            if (state, action) not in all_state_action_pairs[:t]:
                R = sum(rewards[t:])

                total_return[(state, action)] += R
                N[(state, action)] += 1

                Q[(state, action)] = total_return[(state, action)] / N[(state, action)]

        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{num_iterations}, epsilon: {epsilon:.4f}")

    policy = np.zeros((grid_size, grid_size), dtype=int)
    for pos in range(grid_size):
        for vel in range(grid_size):
            state = (pos, vel)
            policy[pos][vel] = max(range(env.action_space.n), key=lambda a: Q[(state, a)])

    return Q, policy


def test(env, optimal_policy, render=True):
    state, _ = env.reset()

    if render:
        env.render()

    total_reward = 0
    while True:
        pos_idx, vel_idx = discretize_state(state)
        action = int(optimal_policy[pos_idx][vel_idx])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if render:
            env.render()

        total_reward += reward
        if done:
            break

    return total_reward


if __name__ == '__main__':
    env = MountainCarEnv()
    env.reset(seed=0)

    Q, optimal_policy = monte_carlo(env, num_iterations=1000, epsilon=0.7, epsilon_decay=0.999)

    print("optimal_policy:\n", optimal_policy)
    env.render_mode = 'human'
    total_reward = test(env, optimal_policy, render=True)
    print(total_reward)

    sum_reward = 0
    for _ in range(3):
        total_reward = test(env, optimal_policy, render=True)
        sum_reward += total_reward

    print("Average reward over 3 episodes: ", sum_reward / 5)
    env.close()
