import gym
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

env = gym.make('SpaceInvaders-v0', render_mode = 'human')
height, width, channels = env.observation_space.shape
actions = env.action_space.n
# action: 0=none, 1=fire, 2=right, 3=left, 4=rightfire, 5=leftfire


# eps = 5
# for eps in range(1,1+eps):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = random.choice([0,1,2,3,4,5])
#         _, reward, done, _ = env.step(action)
#         score += reward
    
#     print(f'Episode: {eps}, Score: {score}')
# env.close()

# Building the model
def build_model(actions,height,width,channels):
    model = Sequential()
    model.add(Flatten(input_shape=(3,height,width,channels)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(actions,height,width,channels)

# Building the agent
def build_agent(model, actions):
    agent = DQNAgent(
        model = model,
        memory = SequentialMemory(limit=1000, window_length=3),
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_min=-1, value_max=1, value_test=0.2, nb_steps=1000),
        nb_actions = actions,
        nb_steps_warmup = 1000
        )
    return agent

agent = build_agent(model,actions)
agent.compile(Adam(lr=0.01))
agent.fit(env, nb_steps=1000, visualize=False, verbose=1)

scores = agent.test(env, nb_episodes=5, visualize=True)
print(np.mean(scores.history['episode_reward']))

