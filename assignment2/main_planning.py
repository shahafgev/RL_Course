import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyRDDLGym

# Ensure script runs as if in its own folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


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
            policy.append(0)  # terminal state, action doesn't matter
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

def state_to_index(state):
    return sum([(int(state[f'job{i+1}']) << i) for i in range(NUM_JOBS)])

def int_to_state(s_idx):
    return [int(bool(s_idx & (1 << i))) for i in range(NUM_JOBS)]

def td0_learn_v_pi_c(env, pi_c, V_pi_c, num_episodes=10000, alpha_schedule=None):
    s0 = 0

    V = np.zeros(NUM_STATES)
    visit_counts = np.zeros(NUM_STATES)
    max_errors = []
    s0_errors = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            s_idx = state_to_index(state)
            action = pi_c[s_idx]
            if action == 0:
                break

            next_state, reward, done, _, _ = env.step({f'do_job{action}': True})
            s_next_idx = state_to_index(next_state)

            visit_counts[s_idx] += 1
            alpha = alpha_schedule(visit_counts[s_idx]) if alpha_schedule else 0.1
            target = reward + (0 if done else V[s_next_idx])
            V[s_idx] += alpha * (target - V[s_idx])

            state = next_state

        # Track errors after each episode
        max_errors.append(np.max(np.abs(V - V_pi_c)))
        s0_errors.append(abs(V[s0] - V_pi_c[s0]))

    return V, max_errors, s0_errors

def q_learning_with_eval(env, V_star, alpha_schedule, 
                         num_episodes=5000, epsilon=0.1, eval_interval=100):
    gamma = 1.0
    s0 = 0

    Q = np.zeros((NUM_STATES, NUM_JOBS))
    visit_counts = np.zeros((NUM_STATES, NUM_JOBS))
    max_errors = []
    s0_errors = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False

        while not done:
            s_idx = state_to_index(state)
            state_bits = int_to_state(s_idx)
            available_actions = [i for i in range(NUM_JOBS) if state_bits[i] == 0]

            if not available_actions:
                break  # terminal state

            # ε-greedy action selection among unfinished jobs
            if np.random.rand() < epsilon:
                action = np.random.choice(available_actions)
            else:
                q_vals = Q[s_idx, available_actions]
                action = available_actions[np.argmin(q_vals)]

            action_dict = {f'do_job{action + 1}': True}  # 1-based job index
            next_state, reward, done, _, _ = env.step(action_dict)
            s_next_idx = state_to_index(next_state)

            visit_counts[s_idx, action] += 1
            alpha = alpha_schedule(visit_counts[s_idx, action])
            best_next_Q = np.min(Q[s_next_idx]) if any(int_to_state(s_next_idx)[i] == 0 for i in range(NUM_JOBS)) else 0
            Q[s_idx, action] += alpha * (reward + gamma * best_next_Q - Q[s_idx, action])

            state = next_state

        # Evaluate greedy policy every eval_interval episodes
        if episode % eval_interval == 0:
            # Extract greedy policy considering only unfinished jobs
            pi_Q = []
            for s_idx in range(NUM_STATES):
                state_bits = int_to_state(s_idx)
                unfinished = [i for i in range(NUM_JOBS) if state_bits[i] == 0]
                if unfinished:
                    best_action = unfinished[np.argmin(Q[s_idx, unfinished])]
                    pi_Q.append(best_action + 1)  # 1-based action
                else:
                    pi_Q.append(0)  # terminal

            V_pi_Q = compute_value_function_iterative(pi_Q, mu, c)
            max_errors.append(np.max(np.abs(V_star - V_pi_Q)))
            s0_errors.append(abs(V_star[s0] - V_pi_Q[s0]))

    return Q, max_errors, s0_errors



c = [1, 4, 6, 2, 9]
mu = [0.6, 0.5, 0.3, 0.7, 0.1]

NUM_STATES=32
NUM_JOBS = len(c)

# Build pi_c and compute its value function
pi_c = build_pi_c(c)
V_pi_c = compute_value_function_iterative(pi_c, mu, c)
optimal_policy, V_optimal, trace = policy_iteration(mu, c, pi_c)

# # Plot the values
# plt.figure(figsize=(10, 5))
# plt.plot(range(32), V_pi_c, marker='o')
# plt.title('Value Function for Policy πc (Highest Cost Job First)')
# plt.xlabel('State index')
# plt.ylabel('V(s)')
# plt.grid(True)
# plt.savefig('plots/value_function_for_policy_pi.png', dpi=300)
# plt.close()

# plt.plot(trace, marker='o')
# plt.title('Convergence of V(0) During Policy Iteration')
# plt.xlabel('Iteration')
# plt.ylabel('V(0)')
# plt.grid(True)
# plt.savefig('plots/convergence_of_v0_policy_iteration.png', dpi=300)
# plt.close()


env = pyRDDLGym.make(domain='job_scheduler_domain.rddl', instance='job_scheduler_instance.rddl')

def alpha_1_over_n(n): return 1.0 / n if n > 0 else 1.0
def alpha_const(n): return 0.01
def alpha_decay(n): return 10.0 / (100 + n)

def plot_errors(max_errors, s0_errors, save_name):
    plt.plot(max_errors, label='||V_pi - V_td||_∞')
    plt.plot(s0_errors, label='|V_pi(s0) - V_td(s0)|')
    plt.xlabel('Episodes')
    plt.ylabel('Error')
    plt.ylim(0)
    plt.legend()
    plt.title('TD(0) Convergence')
    plt.savefig(f'plots/{save_name}.png', dpi=300)
    plt.close()


# V_td, max_errors, s0_errors = td0_learn_v_pi_c(env, pi_c, V_pi_c, alpha_schedule=alpha_1_over_n)
# plot_errors(max_errors, s0_errors, "Errors_1")

# V_td2, max_errors2, s0_errors2 = td0_learn_v_pi_c(env, pi_c, V_pi_c, alpha_schedule=alpha_const)
# plot_errors(max_errors2, s0_errors2, "Errors_2")

# V_td3, max_errors3, s0_errors3 = td0_learn_v_pi_c(env, pi_c, V_pi_c, alpha_schedule=alpha_decay)
# plot_errors(max_errors3, s0_errors3, "Errors_3")

def alpha_cons2(n): return 0.001

# Run Q-learning
Q, max_errors, s0_errors = q_learning_with_eval(
    env=env,
    V_star=V_optimal,
    alpha_schedule=alpha_const,
    eval_interval=10,
    epsilon=0.01)
plot_errors(max_errors, s0_errors, "Q_errors_5")