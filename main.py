import pyRDDLGym
import numpy as np

# Create the environment
myEnv = pyRDDLGym.make(domain='casino_domain.rddl', instance='casino_instance.rddl')

# Reset environment
state, _ = myEnv.reset()

# Manual action sequence: 2, 1, 2
manual_actions = [2, 1, 2]

total_reward = 0
for step, action_value in enumerate(manual_actions):
    # Create an action dictionary with int
    action = {'action': action_value}

    # Step environment
    next_state, reward, done, info, _ = myEnv.step(action)
    total_reward += reward

    print(f'step       = {step}')
    print(f'state      = {state}')
    print(f'action     = {action}')
    print(f'next state = {next_state}')
    print(f'reward     = {reward}\n')

    state = next_state
    if done:
        break

print(f'Episode ended with total reward: {total_reward}')
myEnv.close()
