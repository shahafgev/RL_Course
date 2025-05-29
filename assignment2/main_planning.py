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

def build_pi_c_mu(c, mu):
    policy = []
    for s in range(NUM_STATES):
        state = int_to_state(s)
        unfinished_jobs = [i for i in range(NUM_JOBS) if state[i] == 0]
        if unfinished_jobs:
            best_job = max(unfinished_jobs, key=lambda i: c[i] * mu[i])
            policy.append(best_job + 1)  # 1-based index
        else:
            policy.append(0)  # terminal state
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

def td0_learn_v_pi_c(env, pi_c, V_pi_c, num_episodes=10000, alpha_schedule=None, epsilon=0.1):
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
            state_bits = int_to_state(s_idx)
            available_actions = [i for i in range(NUM_JOBS) if state_bits[i] == 0]

            if not available_actions:
                break  # terminal state

            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(available_actions) + 1  # 1-based
            else:
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
                         num_episodes=10000, epsilon=0.1, eval_interval=100):
    gamma = 1.0

    Q = np.zeros((NUM_STATES, NUM_JOBS))
    visit_counts = np.zeros((NUM_STATES, NUM_JOBS))
    max_errors = []
    s0_errors = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False

        # one episode of Q-learning
        while not done:
            s_idx = state_to_index(state)
            state_bits = int_to_state(s_idx)
            available_actions = [i for i in range(NUM_JOBS) if state_bits[i] == 0]
            if not available_actions:
                break

            # epsilon-greedy among unfinished jobs
            if np.random.rand() < epsilon:
                action = np.random.choice(available_actions)
            else:
                q_vals = Q[s_idx, available_actions]
                action = available_actions[np.argmin(q_vals)]

            action_dict = {f'do_job{action + 1}': True}
            next_state, reward, done, _, _ = env.step(action_dict)
            s_next_idx = state_to_index(next_state)

            # update counts & alpha
            visit_counts[s_idx, action] += 1
            alpha = alpha_schedule(visit_counts[s_idx, action])

            # restrict lookahead to next state's unfinished jobs
            next_bits = int_to_state(s_next_idx)
            next_available_actions = [i for i in range(NUM_JOBS) if next_bits[i] == 0]
            best_next_Q = (np.min(Q[s_next_idx, next_available_actions])
                           if next_available_actions else 0)

            Q[s_idx, action] += alpha * (reward + gamma * best_next_Q - Q[s_idx, action])
            state = next_state

        # evaluate every eval_interval episodes
        if episode % eval_interval == 0:
            # build greedy policy π̂ from Q
            pi_Q = []
            for s in range(NUM_STATES):
                bits_s = int_to_state(s)
                avail_s = [i for i in range(NUM_JOBS) if bits_s[i] == 0]
                if avail_s:
                    best_a = avail_s[np.argmin(Q[s, avail_s])]
                    pi_Q.append(best_a + 1)
                else:
                    pi_Q.append(0)

            # policy-evaluation
            V_pi_Q = compute_value_function_iterative(pi_Q, mu, c)
            max_errors.append(np.max(np.abs(V_star - V_pi_Q)))

            # s0-error at state 0
            s0 = 0
            bits0 = int_to_state(s0)
            avail0 = [i for i in range(NUM_JOBS) if bits0[i] == 0]
            min_Q_s0 = np.min(Q[s0, avail0]) if avail0 else 0
            s0_errors.append(abs(V_star[s0] - min_Q_s0))

    return Q, max_errors, s0_errors

def plot_errors(max_errors, s0_errors, title, label_max='||V_pi - V_td||_∞', label_s0='|V_pi(s0) - V_td(s0)|' ):
    plt.plot(max_errors, label=label_max)
    plt.plot(s0_errors, label=label_s0)
    plt.xlabel('Episodes')
    plt.ylabel('Error')
    plt.ylim(0)
    plt.legend()
    plt.title(title)
    plt.savefig(f'plots/{title}.png', dpi=300)
    plt.close()



c = [1, 4, 6, 2, 9]
mu = [0.6, 0.5, 0.3, 0.7, 0.1]

NUM_STATES=32
NUM_JOBS = len(c)

# ---------------- Planning ----------------
# Build pi_c and compute its value function
pi_c = build_pi_c(c)
V_pi_c = compute_value_function_iterative(pi_c, mu, c)

# Compute the optimal policy and value function using policy iteration
optimal_policy, V_optimal, trace = policy_iteration(mu, c, pi_c)
print("The optimal policy is:\n", optimal_policy)
print("The optimal value function is:\n", V_optimal)

# Plot value function by state for policy Pi_c
plt.figure(figsize=(10, 5))
plt.plot(range(32), V_pi_c, marker='o')
plt.title('Value Function for Policy πc (Highest Cost Job First)')
plt.xlabel('State index')
plt.ylabel('V(s)')
plt.grid(True)
plt.savefig('plots/value_function_for_policy_pi.png', dpi=300)
plt.close()

# Plot V(0) during policy iteration
plt.plot(trace, marker='o')
plt.title('Convergence of V(0) During Policy Iteration')
plt.xlabel('Iteration')
plt.ylabel('V(0)')
plt.grid(True)
plt.savefig('plots/convergence_of_v0_policy_iteration.png', dpi=300)
plt.close()

# Compute c-mu policy and value function
pi_c_mu = build_pi_c_mu(c, mu)
V_pi_c_mu = compute_value_function_iterative(pi_c_mu, mu, c)
print("The c-mu policy is:\n", pi_c_mu)
print("The c-mu value function is:\n", V_pi_c_mu)

# Compare both
print("-------------------------------------------------------")
print("Are the two policies the same? ", optimal_policy==pi_c_mu)
print("Are their value function the same? ", V_optimal==V_pi_c_mu)

# Plot V_pi_c vs V_pi*
plt.figure(figsize=(10, 5))
plt.plot(range(32), V_pi_c, marker='o', label="V_π_c")
plt.plot(range(32), V_optimal, marker='X', label="V_optimal")
plt.title('Value Function comparison for Policy πc vs. the optimal policy')
plt.xlabel('State index')
plt.ylabel('V(s)')
plt.legend()
plt.grid(True)
plt.savefig('plots/value_function_comparison.png', dpi=300)
plt.close()

# ---------------- Learning ----------------
env = pyRDDLGym.make(domain='job_scheduler_domain.rddl', instance='job_scheduler_instance.rddl')

def alpha_1_over_n(n): return 1.0 / n if n > 0 else 1.0
def alpha_const(n): return 0.01
def alpha_decay(n): return 10.0 / (100 + n)

# ------ TD(0) ------
V_td, max_errors, s0_errors = td0_learn_v_pi_c(env, pi_c, V_pi_c, alpha_schedule=alpha_1_over_n)
plot_errors(max_errors, s0_errors, title = 'TD(0) Convergence with 1 over n Schedule')

V_td2, max_errors2, s0_errors2 = td0_learn_v_pi_c(env, pi_c, V_pi_c, alpha_schedule=alpha_const)
plot_errors(max_errors2, s0_errors2, title = 'TD(0) Convergence with Constant Schedule')

V_td3, max_errors3, s0_errors3 = td0_learn_v_pi_c(env, pi_c, V_pi_c, alpha_schedule=alpha_decay)
plot_errors(max_errors3, s0_errors3, title = 'TD(0) Convergence with Decay Schedule')


# ------ Q-learning ------
Q, max_errors, s0_errors = q_learning_with_eval(env=env, V_star=V_optimal, alpha_schedule=alpha_1_over_n, eval_interval=10)
plot_errors(max_errors, s0_errors, title = 'Q-learning Convergence with 1 over n Schedule', label_max='||V* - V_pi_Q||_∞', label_s0='|V*(s0) - argmin Q(s0,a)|')

Q, max_errors, s0_errors = q_learning_with_eval(env=env, V_star=V_optimal, alpha_schedule=alpha_const)
plot_errors(max_errors, s0_errors, title = 'Q-learning Convergence with Constant Schedule', label_max='||V* - V_pi_Q||_∞', label_s0='|V*(s0) - argmin Q(s0,a)|')

Q, max_errors, s0_errors = q_learning_with_eval(env=env, V_star=V_optimal, alpha_schedule=alpha_decay)
plot_errors(max_errors, s0_errors, title = 'Q-learning Convergence with Decay Schedule', label_max='||V* - V_pi_Q||_∞', label_s0='|V*(s0) - argmin Q(s0,a)|')

# Now with epsilon 0.01
Q, max_errors, s0_errors = q_learning_with_eval(env=env, V_star=V_optimal, alpha_schedule=alpha_decay, epsilon=0.01)
plot_errors(max_errors, s0_errors, title = 'Q-learning Convergence with Decay Schedule (epsilon 0.01)', label_max='||V* - V_pi_Q||_∞', label_s0='|V*(s0) - argmin Q(s0,a)|')

