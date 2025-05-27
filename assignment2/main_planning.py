import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyRDDLGym

# Ensure script runs as if in its own folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

env = pyRDDLGym.make(domain='job_schedualer.rddl', instance='instance_5jobs.rddl')

def int_to_state(i):
    return [(i >> j) & 1 for j in range(5)]  # least significant bit first

def state_to_int(state):
    return sum([(bit << i) for i, bit in enumerate(state)])

def state_cost(state, c):
    return sum([c[i] for i in range(5) if state[i] == 0])

def next_state(state, job_index):
    new_state = state[:]
    new_state[job_index] = 1
    return new_state

def compute_value_function_iterative(policy, mu, c, tol=1e-6, max_iter=1000):
    V = np.zeros(NUM_STATES)

    for _ in range(max_iter):
        V_new = np.zeros_like(V)
        delta = 0

        for s in range(NUM_STATES):
            state = int_to_state(s)

            if all(x == 1 for x in state):  # terminal state
                V_new[s] = 0
                continue

            job = policy[s] - 1  # 1-based to 0-based
            if state[job] == 1:
                raise ValueError(f"Invalid policy: job {job+1} already finished in state {s}")

            next_s = state_to_int(next_state(state, job))
            cost = state_cost(state, c)
            mu_i = mu[job]

            # Bellman update
            V_new[s] = cost + mu_i * V[next_s] + (1 - mu_i) * V[s]

            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < tol:
            break

    return V

def build_pi_c(c):
    policy = []
    for s in range(NUM_STATES):
        state = int_to_state(s)
        unfinished_jobs = [i for i in range(NUM_JOBS) if state[i] == 0]
        if unfinished_jobs:
            best_job = max(unfinished_jobs, key=lambda i: c[i])
            policy.append(best_job + 1)  # 1-based index
        else:
            policy.append(1)  # terminal state, action doesn't matter
    return policy

def policy_iteration(mu, c, initial_policy):
    policy = initial_policy[:]
    value_trace = []

    iteration = 0
    while True:
        # Policy evaluation
        V = compute_value_function_iterative(policy, mu, c)
        value_trace.append(V[0])  # Track value at state 0

        # Policy improvement
        policy_stable = True
        new_policy = policy[:]

        for s in range(NUM_STATES):
            state = int_to_state(s)
            if all(x == 1 for x in state):  # terminal state
                continue

            best_action = None
            best_value = float('inf')

            for a in range(NUM_JOBS):
                if state[a] == 1:
                    continue  # skip finished jobs

                next_s = state_to_int(next_state(state, a))
                cost = state_cost(state, c)
                mu_a = mu[a]

                expected_value = cost + mu_a * V[next_s] + (1 - mu_a) * V[s]

                if expected_value < best_value:
                    best_value = expected_value
                    best_action = a + 1  # 1-based

            if best_action != policy[s]:
                new_policy[s] = best_action
                policy_stable = False

        policy = new_policy
        iteration += 1

        if policy_stable:
            break

    V_final = compute_value_function_iterative(policy, mu, c)
    return policy, V_final, value_trace

def td0_learn_v_pi_c(env, pi_c, num_episodes=5000, alpha_schedule=None):
    V = np.zeros(NUM_STATES)
    visit_counts = np.zeros(NUM_STATES)

    # Converts simulator state to an integer index (0–31) based on job_done bits.
    def state_to_index(state):
        return sum([(int(state[f'job_done(j{i+1})']) << i) for i in range(NUM_JOBS)])

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            s_idx = state_to_index(state)
            visit_counts[s_idx] += 1

            # Choose action from policy π_c
            unfinished = [i for i in range(NUM_JOBS) if not state[f'job_done(j{i+1})']]
            if not unfinished:
                break

            action = pi_c(unfinished)
            next_state, cost, done = env.step(action)
            s_next_idx = state_to_index(next_state)

            # Choose alpha (step size)
            alpha = alpha_schedule(visit_counts[s_idx]) if alpha_schedule else 0.1

            # TD(0) update
            target = cost + (0 if done else V[s_next_idx])
            V[s_idx] += alpha * (target - V[s_idx])

            state = next_state

    return V



c = [1, 4, 6, 2, 9]
mu = [0.6, 0.5, 0.3, 0.7, 0.1]

NUM_STATES=32
NUM_JOBS = len(c)

# Build pi_c and compute its value function
pi_c = build_pi_c(c)
V_pi_c = compute_value_function_iterative(pi_c, mu, c)
optimal_policy, V_optimal, trace = policy_iteration(mu, c, pi_c)

# Plot the values
plt.figure(figsize=(10, 5))
plt.plot(range(32), V_pi_c, marker='o')
plt.title('Value Function for Policy πc (Highest Cost Job First)')
plt.xlabel('State index')
plt.ylabel('V(s)')
plt.grid(True)
plt.savefig('plots/value_function_for_policy_pi.png', dpi=300)
plt.close()

plt.plot(trace, marker='o')
plt.title('Convergence of V(0) During Policy Iteration')
plt.xlabel('Iteration')
plt.ylabel('V(0)')
plt.grid(True)
plt.savefig('plots/convergence_of_v0_policy_iteration.png', dpi=300)
plt.close()