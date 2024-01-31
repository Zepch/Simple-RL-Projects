import gym
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Creating the environment
env = gym.make("CartPole-v1")

# Building the model
states = env.observation_space.shape[0]
actions = env.action_space.n

def buildmodel():
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation = 'linear'))
    return model

# Building the agent
agent = DQNAgent(
    model=buildmodel(),
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

agent.compile(Adam(learning_rate=0.001), metrics=['mae'])

agent.fit(env, nb_steps=50000, visualize=True, verbose=1)
results = agent.test(env, nb_episodes=10, visualize=True)

print(np.mean(results.history['episode_reward']))

# eps = 10
# for i in range(1, eps+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = random.choice([0,1])
#         _, reward, done, _ = env.step(action)
#         score += reward

#     print(f'Episode {i}, score: {score}')