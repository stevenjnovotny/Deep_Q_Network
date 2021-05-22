
# Navigation Using Deep Reinforcement Learning

## Introduction

This project uses a modified version the 'Banana' environemnt from Unity's ml-agents to demonstrate deep reinforcement learning in an AI agent. 

<p align="center">
  <img src=environment.png#center />
</p>

A description of the Unity environments can be found here:

https://github.com/Unity-Technologies/ml-agents/tree/main/ml-agents-envs/mlagents_envs


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.  The first 35 elements represent seven rays directed at 20, 90, 160, 45, 135, 70, 110 degrees with 90 being directly in front of the agent. For each ray, a five element vector is used to indicate presence of a banana, a wall, a bad banana, another agent, and a distance. Finally, there are two elements for lateral and transverse velocity.

Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and the environment is considered solved when the agent achieves an average score of +13 over 100 consecutive episodes.

### Using the Envoironment

The environment can be downloaded for MacOS from the following link:
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)


## Instructions for Running the Agent

To run the simulation, run `Navigation.py`.
- You will first see the results for an untrained agent, taking random actions
- Then, a deep q-learning agent will be trained on the environment.
- Finally, the agent's score as a function of epoch will be displayed. An example is shown below.


<p align="center">
  <img src=Figure_1.png#center width="350"/>
</p>


## The Agent and the Model

The agent was developed and trained in the following way:
- A neural network architecture was defined in `model.py` to map states to action values. It consists of 37 inputs, two hidden layers of 64 neurons each, and 4 outputs. This represents a standard states-in-values-out network.

- The agent itself is defined in the `rl_agent.py` file. The following methods are key to the agent's learning/behavior and are described within the code:
  - `step`(state, action, reward, next_state, done)
  - `get_action`(state, eps=0.0)  
  - `learn`(experiences, gamma) 
  - `soft_update`(local_model, target_model, tau)

- The `rl_agent.py` file defines the agent. This agent uses a double deep q-network (DDQN) where theta represents the local weights and theta prime the target weights. This means the next action is selected from the local (online) newtork and evaluated with the frozen (target) network. This learning is represented by the following update equation expressing the double q-learning error:
<p align="center">
  <img src=double_dqn.png#center width="350"/>
</p>


- In addition, the agent uses experience replay to break harmful correlations and make repeated use of experiences. In this case the buffer holds 1e5 experience tuples and learns in batches of 64. The experience replay is also described in the `rl_agent.py` file.


## Improvements

The following techniques could be used to improve learning efficency:
- Prioritized experience replay: This is based on evaluating which experiences are more impactful to the learning process and biasing the selction process to select those experiences.
- Dueling DQN: This technique uses dueling networks to evaluate q-values and advantage values seperately.


## References:
- Morales, M. (2020) "Grokking Deep Reinforcement Learning." Manning Publications.
-  Van Hasselt, Hado & Guez, Arthur & Silver, David. (2015). "Deep Reinforcement Learning with Double Q-learning." Google DeepMind. 