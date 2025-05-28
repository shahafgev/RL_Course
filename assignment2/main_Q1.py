import os
import random
import numpy as np
import pyRDDLGym
import matplotlib.pyplot as plt

# Ensure script runs as if in its own folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def run_random_policy(env):
    regret = []
    obs, done = env.reset(), False
    cumulative_regret = 0
    
    while not done:
        arm = random.choice(ARMS)
        action = {f'roll___{a}': a == arm for a in ARMS}
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_regret += mu_star - reward
        regret.append(cumulative_regret)

    # print("Finished random")
    return regret

def run_greedy_policy(env):
    arm_rewards = {a: [] for a in ARMS}
    obs, done = env.reset(), False
    cumulative_regret = 0
    regret = []

    # Explore each arm 100 times
    for a in ARMS:
        for _ in range(100):
            action = {f'roll___{a}': True}
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            arm_rewards[a].append(reward)
            cumulative_regret += mu_star - reward
            regret.append(cumulative_regret)

    estimated_means = {a: np.mean(arm_rewards[a]) for a in ARMS}
    best_arm = max(estimated_means, key=estimated_means.get)

    while not done:
        action = {f'roll___{a}': a == best_arm for a in ARMS}
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_regret += mu_star - reward
        regret.append(cumulative_regret)
    # print("Finished greedy")
    return regret

def run_ucb1_policy(env):
    n = len(ARMS)
    counts = np.zeros(n)
    values = np.zeros(n)
    regret = []
    cumulative_regret = 0
    obs, done = env.reset(), False
    t = 0 # Turn index
    while not done:
        if t < n: 
            arm_idx = t # Set the arm to roll as the turn index
        else:
            upper_bounds = values + np.sqrt(2 * np.log(t + 1) / (counts + 1e-9))
            arm_idx = np.argmax(upper_bounds)

        arm = ARMS[arm_idx]
        action = {f'roll___{a}': a == arm for a in ARMS}
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        counts[arm_idx] += 1
        values[arm_idx] += (reward - values[arm_idx]) / counts[arm_idx] # Incremental mean update

        cumulative_regret += mu_star - reward
        regret.append(cumulative_regret)
        t += 1

    # print("Finished ucb")
    return regret


domain_file = "MAB_domain.rddl"
instance_file = "MAB_instance.rddl"
env = pyRDDLGym.make(domain=domain_file, instance=instance_file)

NUM_ARMS = 100
EPISODES = 20
ARMS = [a.split('___')[1] for a in env.action_space.keys() if a.startswith('roll___')]

# True mean rewards
arm_probs = [(i + 1) / (NUM_ARMS + 1) for i in range(NUM_ARMS)]
mu_star = max(arm_probs)  # True best expected reward

random_regrets = []
greedy_regrets = []
ucb1_regrets = []

for _ in range(EPISODES):
    random_regrets.append(run_random_policy(env))
    greedy_regrets.append(run_greedy_policy(env))
    ucb1_regrets.append(run_ucb1_policy(env))

avg_random = np.mean(random_regrets, axis=0)
avg_greedy = np.mean(greedy_regrets, axis=0)
avg_ucb1 = np.mean(ucb1_regrets, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(avg_random, label='Random')
plt.plot(avg_greedy, label='Greedy')
plt.plot(avg_ucb1, label='UCB1')
plt.xlabel('Timestep')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.title('Bandit Algorithms Regret Comparison (Averaged over 20 Runs)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'plots/bandit.png', dpi=300)
plt.close()