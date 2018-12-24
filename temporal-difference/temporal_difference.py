import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')
print(env.action_space)
print(env.observation_space)

# define the optimal state-value function
V_opt = np.zeros((4,12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

plot_values(V_opt)


def eps_greedy_act(p_q_state, p_env, p_eps):
    greed = np.random.choice(np.arange(2), p=[p_eps, 1-p_eps])
    if greed:
        action = np.argmax(p_q_state)
    else:
        action = np.random.randint(0, p_env.action_space.n-1)
    return action


def sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            # plot the estimated optimal state-value function
            V_sarsa = ([np.max(Q[key]) if key in Q else 0 for key in np.arange(48)])
            plot_values(V_sarsa)

        s0 = env.reset()
        """
        Q[s0, a0] = (1-alpha) * Q[s0, a0] + alpha * (r + gamma * Q[s1,a1])
        """
        a0 = eps_greedy_act(Q[s0], env, 1.0/i_episode)

        for i in range(1000):
            [s1, r, done, info] = env.step(a0)
            if not done:
                a1 = eps_greedy_act(Q[s1], env, 1.0/i_episode)
                Q[s0][a0] = (1 - alpha) * Q[s0][a0] + alpha * (r + gamma * Q[s1][a1])
            else:
                Q[s0][a0] = (1 - alpha) * Q[s0][a0] + alpha * r
                break
            a0 = a1
            s0 = s1

    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 300, .1)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)


def q_learning(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 20 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            # plot the estimated optimal state-value function
            #V_q = ([np.max(Q[key]) if key in Q else 0 for key in np.arange(48)])
            #plot_values(V_q)

        s0 = env.reset()

        for i in range(1000):
            a = eps_greedy_act(Q[s0], env, 1.0 / i_episode)
            [s1, r, done, info]=env.step(a)
            if not done:
                a_max = eps_greedy_act(Q[s1],env, 0)
                Q[s0][a] = (1 - alpha) * Q[s0][a] + alpha * (r + gamma * Q[s1][a_max])
            else:
                Q[s0][a] = (1 - alpha) * Q[s0][a] + alpha * r
                break
            s0 = s1
    return Q

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsamax = q_learning(env, 100, .1)

# print the estimated optimal policy
policy_sarsamax = np.array(
    [np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4, 12))
check_test.run_check('td_control_check', policy_sarsamax)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# plot the estimated optimal state-value function
plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])


def eps_greedy_p(p_q_state, p_env, p_eps):
    p= np.ones(p_env.action_space.n)*p_eps/p_env.action_space.n
    p[np.argmax(p_q_state)] += 1-p_eps
    return p

def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 20 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            # plot the estimated optimal state-value function
            #V_q = ([np.max(Q[key]) if key in Q else 0 for key in np.arange(48)])
            #plot_values(V_q)

        s0 = env.reset()

        for i in range(1000):
            a = eps_greedy_act(Q[s0], env, 1.0 / i_episode)
            [s1, r, done, info]=env.step(a)
            if not done:
                p = eps_greedy_p(Q[s1],env, 1.0 / i_episode)
                Q[s0][a] = (1 - alpha) * Q[s0][a] + alpha * (r + gamma * np.sum(Q[s1]*p))
            else:
                Q[s0][a] = (1 - alpha) * Q[s0][a] + alpha * r
                break
            s0 = s1
    return Q

# obtain the estimated optimal policy and corresponding action-value function
Q_expsarsa = expected_sarsa(env, 300, 0.1)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
