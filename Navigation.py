from numpy.core.fromnumeric import _diagonal_dispatcher
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# start the environment
env = UnityEnvironment(file_name="Banana.app")
print(env)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

'''
The simulation contains a single agent that navigates a large environment. At each 
time step, it has four actions at its disposal:

0 - walk forward
1 - walk backward
2 - turn left
3 - turn right

The state space has 37 dimensions and contains the agent's velocity, along 
with ray-based perception of objects around agent's forward direction. A reward 
of +1 is provided for collecting a yellow banana, and a reward of -1 is provided 
for collecting a blue banana.

state space:  
7 rays : [20, 90, 160, 45, 135, 70, 110] # 90 is directly in front of the agent
-- 5 element vector for each ray : [Banana, Wall, BadBanana, Agent, Distance]
2 elements for velocity

print some information about the environment.
'''

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)



'''
Take Random Actions in the Environment

receive feedback from the environment.

A window should pop up that allows you to observe the agent, as it moves through the environment.

use its experience to gradually choose better actions when interacting 
with the environment!

'''

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
for i in range(20):
    # print('----------------------')
    # print(env_info.vector_observations)

    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step

    # print(action)
    # print(reward)
    # print(done)

    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))


from rl_agent import Agent
import torch
agent = Agent(state_size=state_size, num_actions=action_size, seed=0)

def train_dqn(n_episodes=1000, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    '''
    train a deep q network
    '''

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1,n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0] 
        score = 0
        for t in range(max_t):
            action = agent.get_action(state, eps)          # get action from agent
            env_info = env.step(action)[brain_name]        
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))  
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoints.pth')    
    
    return scores  

scores = train_dqn()
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()