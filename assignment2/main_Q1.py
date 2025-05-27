import os
import random
import numpy as np
import pyRDDLGym
import matplotlib.pyplot as plt

# Ensure script runs as if in its own folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def run_random_policy(env, horizon):
    regret = []
    obs, done = env.reset(), False
    cumulative_regret = 0

    for t in range(horizon):
        arm = random.choice(env.action_space.param_space['drawn'])
        action = {f'drawn({a})': a == arm for a in env.action_space.param_space['drawn']}
        obs, reward, done, _ = env.step(action)
        cumulative_regret -= reward
        regret.append(cumulative_regret)

    return regret

def run_greedy_policy(env, horizon):
    arm_rewards = {a: [] for a in env.action_space.param_space['drawn']}

    # Explore each arm 100 times
    for a in arm_rewards:
        for _ in range(100):
            env.reset()
            action = {f'drawn({x})': x == a for x in arm_rewards}
            obs, reward, done, _ = env.step(action)
            arm_rewards[a].append(-reward)

    best_arm = max(arm_rewards, key=lambda a: np.mean(arm_rewards[a]))

    obs, done = env.reset(), False
    cumulative_regret = 0
    regret = []
    for t in range(horizon):
        action = {f'drawn({a})': a == best_arm for a in arm_rewards}
        obs, reward, done, _ = env.step(action)
        cumulative_regret -= reward
        regret.append(cumulative_regret)

    return regret

def run_ucb1_policy(env, horizon):
    n = len(env.action_space.param_space['drawn'])
    counts = np.zeros(n)
    values = np.zeros(n)
    arms = list(env.action_space.param_space['drawn'])
    regret = []
    cumulative_regret = 0

    for t in range(horizon):
        if t < n:
            arm_idx = t
        else:
            upper_bounds = values + np.sqrt(2 * np.log(t + 1) / (counts + 1e-9))
            arm_idx = np.argmax(upper_bounds)

        arm = arms[arm_idx]
        action = {f'drawn({a})': a == arm for a in arms}
        obs, reward, done, _ = env.step(action)
        x = -reward
        counts[arm_idx] += 1
        values[arm_idx] += (x - values[arm_idx]) / counts[arm_idx] 

        cumulative_regret += x
        regret.append(cumulative_regret)

    return regret

if __name__ == '__main__':
    domain_file = "BanditSim.rddl"
    instance_file = "BanditSim_100.rddl"
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file)

    horizon = 20000
    episodes = 1

    avg_random = np.mean([run_random_policy(env, horizon) for _ in range(episodes)], axis=0)
    avg_greedy = np.mean([run_greedy_policy(env, horizon) for _ in range(episodes)], axis=0)
    avg_ucb1 = np.mean([run_ucb1_policy(env, horizon) for _ in range(episodes)], axis=0)

    plt.plot(avg_random, label='Random')
    plt.plot(avg_greedy, label='Greedy')
    plt.plot(avg_ucb1, label='UCB1')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.title('Bandit Algorithms Regret Comparison')
    plt.grid(True)
    plt.show()
