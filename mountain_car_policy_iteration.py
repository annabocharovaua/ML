import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

grid_size = 15
position_grid = np.linspace(-1.2, 0.6, grid_size)
velocity_grid = np.linspace(-0.07, 0.07, grid_size)


def discretize_state(state):
    position_index = np.digitize(state[0], position_grid) - 1
    velocity_index = np.digitize(state[1], velocity_grid) - 1
    position_index = max(0, min(position_index, grid_size - 1))
    velocity_index = max(0, min(velocity_index, grid_size - 1))
    return position_index, velocity_index

def compute_value_function(env, policy, num_iterations=1000, threshold=1e-20, gamma=0.99):
    value_table = np.zeros((grid_size, grid_size))

    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)

        for pos in range(grid_size):
            for vel in range(grid_size):
                action = int(policy[pos][vel])

                env.state = np.array([position_grid[pos], velocity_grid[vel]])
                prev_state = env.state.copy()
                next_state, reward, terminated, truncated, info = env.step(action)
                reward = reward + np.abs(next_state[0] - prev_state[0])

                value_table[pos][vel] = reward + gamma * updated_value_table[discretize_state(next_state)]

        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break

    return value_table


def extract_policy(env, value_table, gamma=0.99):
    policy = np.zeros((grid_size, grid_size), dtype=int)

    for pos in range(grid_size):
        for vel in range(grid_size):
            q_values = []
            for action in range(env.action_space.n):
                env.state = np.array([position_grid[pos], velocity_grid[vel]])
                prev_state = env.state.copy()
                next_state, reward, terminated, truncated, info = env.step(action)
                reward = reward + np.abs(next_state[0] - prev_state[0])
                q_value = reward + gamma * value_table[discretize_state(next_state)]
                q_values.append(q_value)

            policy[pos][vel] = np.argmax(q_values)

    return policy


def policy_iteration(env, num_iterations=10, gamma=0.99):
    policy = np.zeros((grid_size, grid_size), dtype=int)

    for i in range(num_iterations):
        value_table = compute_value_function(env, policy, gamma=gamma)

        new_policy = extract_policy(env, value_table, gamma=gamma)

        if np.all(policy == new_policy):
            print(f"Policy converged at iteration {i + 1}")
            break

        policy = new_policy

    return value_table, policy


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
    state = env.reset(seed=0)
    optimal_value_function, optimal_policy = policy_iteration(env=env)

    print("optimal_policy:\n", optimal_policy)
    env.render_mode = 'human'
    total_reward = test(env, optimal_policy)
    print(total_reward)

    sum_reward = 0
    for _ in range(3):
        total_reward = test(env, optimal_policy, render=True)
        sum_reward += total_reward

    print("Average reward over 3 episodes: ", sum_reward / 5)
    env.close()
