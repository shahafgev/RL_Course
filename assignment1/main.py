import pyRDDLGym
import os
import numpy as np
import matplotlib.pyplot as plt


# Optimal actions by step and state (hardcoded from earlier analysis)
optimal_policy = {
    0: {0: 2},
    1: {0: 2, 1: 1, 2: 1},
    2: {0: 2, 1: 1, 2: 1}
}

# Policy A: Fixed sequence 2,1,2
def policy_fixed_order(state, t):
    return 2 if t == 0 or t == 2 else 1

# Policy B: Random
def policy_random(state, t):
    return np.random.choice([1, 2])

# Policy C: Optimal
def policy_optimal(state, t):
    s = state['S']
    return optimal_policy[t].get(s, 1)  # default to 1 if not specified


def run_trials(policy_func, num_trials):
    rewards = []
    for _ in range(num_trials):
        env = pyRDDLGym.make(domain='casino_domain.rddl', instance='casino_instance.rddl')
        state, _ = env.reset()
        total_reward = 0
        horizon = env.horizon  # get from environment

        for t in range(horizon):
            action = {'action': policy_func(state, t)}
            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        env.close()
        rewards.append(total_reward)
    return np.mean(rewards)



# Simulation range
trial_counts = list(range(1, 1001, 10))
rewards_fixed, rewards_random, rewards_optimal = [], [], []

for n in trial_counts:
    print(f"Started with {n} trials")
    rewards_fixed.append(run_trials(policy_fixed_order, n))
    rewards_random.append(run_trials(policy_random, n))
    rewards_optimal.append(run_trials(policy_optimal, n))
    print(f"Finished with {n} trials")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(trial_counts, rewards_fixed, label="Policy A: Fixed 2-1-2", marker='o')
plt.plot(trial_counts, rewards_random, label="Policy B: Random", marker='s')
plt.plot(trial_counts, rewards_optimal, label="Policy C: Optimal", marker='^')
plt.xlabel('Number of Simulations')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Number of Simulations')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plots/policy_comparison.png', dpi=300)
plt.close()

