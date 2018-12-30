#!/usr/bin/env python
# coding: utf-8

# # Tile Coding
# ---
#
# Tile coding is an innovative way of discretizing a continuous space that enables better generalization compared to a single grid-based approach. The fundamental idea is to create several overlapping grids or _tilings_; then for any given sample value, you need only check which tiles it lies in. You can then encode the original continuous value by a vector of integer indices or bits that identifies each activated tile.
#
# ### 1. Import the Necessary Packages

# In[1]:


# Import common libraries
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt


from tile_encode import tile_encode, create_tilings
from tile_visualize import visualize_tilings, visualize_encoded_samples
from TiledQTable import TiledQTable

# Set plotting options
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

# Create an environment
env = gym.make('Acrobot-v1')
env.seed(505)

# Explore state (observation) space
print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

# Explore action space
print("Action space:", env.action_space)


# 3. Tiling


low = [-1.0, -5.0]
high = [1.0, 5.0]

# Tiling specs: [(<bins>, <offsets>), ...]
tiling_specs = [((10, 10), (-0.066, -0.33)),
                ((10, 10), (0.0, 0.0)),
                ((10, 10), (0.066, 0.33))]
tilings = create_tilings(low, high, tiling_specs)

visualize_tilings(tilings)

# 4. Tile Encoding

# Test with some sample values
samples = [(-1.2 , -5.1 ),
           (-0.75,  3.25),
           (-0.5 ,  0.0 ),
           ( 0.25, -1.9 ),
           ( 0.15, -1.75),
           ( 0.75,  2.5 ),
           ( 0.7 , -3.7 ),
           ( 1.0 ,  5.0 )]
encoded_samples = [tile_encode(sample, tilings) for sample in samples]
print("\nSamples:", repr(samples), sep="\n")
print("\nEncoded samples:", repr(encoded_samples), sep="\n")

visualize_encoded_samples(samples, encoded_samples, tilings)

# 5. Q-Table with Tile Coding

tq = TiledQTable(low, high, tiling_specs, 2)
s1 = 3; s2 = 4; a = 0; q = 1.0
print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value at sample = s1, action = a
print("[UPDATE] Q({}, {}) = {}".format(samples[s2], a, q)); tq.update(samples[s2], a, q)  # update value for sample with some common tile(s)
print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value again, should be slightly updated


# 6. Implement a Q-Learning Agent using Tile-Coding


# Recap on Q-learning:
#   - 0. start with aa blank Qtable
#        for each episode:
#       - a start at the initial state0
#         ->- b choose an epsilon-greedy action0
#         | - c move to a new state1 with a reward
#         | - d from the new state1, identify the greedy actionG
#         | - e update Q table for state0 action0 according to reward and Q table value in state1 actionG
#         <-- f forget actionG, state1 is your new state0, go back to b

# close environment if open
env.close()
# start and seed
env = gym.make('Acrobot-v1')
env.seed(505)



# 0. prepare Q table
print(env.observation_space.high)
print(env.action_space)
q_table = TiledQTable(env.observation_space.low, env.observation_space.high, None, env.action_space.n, 3, 10)
print(q_table.get_tiling_specs())
episode_rewards =[]
average_rewards = 0
num_episodes = 10000
for episode_i in range(num_episodes):
    # a. get starting state
    state0 = env.reset()
    # preparation to step b
    q_values_state = np.array([q_table.get(state0, act) for act in range(env.action_space.n)])
    done = False
    epsilon = 1/(episode_i+5)
    cum_reward = 0

    #while done is False:
    # help avoinding getting stuck on an episode
    for i in range(10000):
        #b. choose an epsilon-greedy action

        take_greedy = (np.random.rand() >= epsilon)
        if take_greedy:
            action0 = np.argmax(q_values_state)
        else:
            action0 = np.random.randint(0, env.action_space.n)

        #c. apply action0 to environment, obtain state1 with a reward
        [state1, reward, done, info] = env.step(action0)
        cum_reward = cum_reward + reward

        if done:
            q_table.update(state0, action0, reward)
            break

        #d. from the new state1, identify the greedy actionG
        q_values_state = np.array([q_table.get(state1, act) for act in range(env.action_space.n)])
        actionG = np.argmax(q_values_state)

        #e. update Q table for state0 action0 according to reward and Q table value in state1 actionG
        q_value_G = q_values_state[actionG]
        q_table.update(state0,action0, reward + q_value_G)

        #f. forget actionG, state1 is your new state0, go back to b
        state0 = state1
    episode_rewards.append(cum_reward)

    av_update= max(1.0/(episode_i+1.0), 0.03)
    average_rewards = average_rewards * (1.0-av_update) + cum_reward *av_update
    if episode_i % 20 == 0:
        print("\rEpisode {}/{}, average reward = {}".format(episode_i, num_episodes, average_rewards), end="")
        sys.stdout.flush()