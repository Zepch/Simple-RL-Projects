import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode = 'human')
env.reset()

learning_rate = 0.1
discount = 0.95
episode = 25000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # Seperate into 20 discrete values
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

discrete_state = get_discrete_state(env.reset())


done = False

while not done:
    action = np.argmax(q_table[discrete_state]) # Move right
    observation, r, done, _ = env.step(action)
    new_discrete_state = get_discrete_state(observation)
    
    if not done:
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]

        new_q = (1-learning_rate) * current_q + learning_rate * (r+discount*max_future_q) # formula for calculating all q values

        q_table[discrete_state + (action, )] = new_q # update current with new qtable
    elif observation[0] >= env.goal_position:
        q_table[discrete_state + (action,)] = 0


env.close()