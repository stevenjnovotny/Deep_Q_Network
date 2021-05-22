from os.path import expanduser
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)      # size of replay buffer
BATCH_SIZE = 64             # size of minibatches for training
GAMMA = 0.99                # discout parameter
TAU = 1e-3                # update parameter for local -> target  (low means slow change to target)
LR = 5e-4                 # learning rate
UPDATE_EVERY = 4            # frequency of updates
QNN_LAYERS = [64,64]        # size of hidden layers in q-network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, num_actions, seed):

        '''
        Initialize an agent

        ---
            state_size: dimensionality of state vector
            num_actions: number of passible actions
            seed: random seed
        ---

        '''

        self.state_size = state_size
        self.num_actions = num_actions
        self.seed = random.seed(seed)

        # create q networks
        self.qnetwork_local = QNetwork(state_size, num_actions, seed, QNN_LAYERS).to(device)
        self.qnetwork_target = QNetwork(state_size, num_actions, seed, QNN_LAYERS).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # set up replay buffer
        self.memory = ReplayBuffer(num_actions, BUFFER_SIZE, BATCH_SIZE, seed)

        # track time
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        '''
        Advance agent; save ecperiences in replay buffer

        learn according to UPDATE_EVERY
        
        '''
        #state = state.float()
        self.memory.add_exp(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # if enough expereinces have been stored, update network
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def get_action(self, state, eps=0.0):

        '''
        returns action according to policy; does not update the local network

        ---
            state: current state
            eps: epsilon for epsilon greedy policy
        ---
        
        '''
        #state = torch.from_numpy(state).unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()  # turn off updating
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()  # turn back on updating

        # handle epsilon greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else: 
            return random.choice(np.arange(self.num_actions))

    def learn(self, experiences, gamma):

        '''
        update value parameters

        ---
            expereinces: (Tuple[torch.Tensor]): tuple of (s,a,r,s',done)
            gamma: discount factor
        ---
        
        '''

        states, actions, rewards, next_actions, dones = experiences
        #states = states.float()

        # get max predicted Q values for next states from target model  (remember: final state value = 0)
        Q_targets_next = self.qnetwork_target(next_actions).detach().max(1)[0].unsqueeze(1)
        # compute Q target values for current state
        Q_targets = rewards + (gamma * Q_targets_next  * (1-dones) )

        # get q values using local; (current) model
        Q_expected = self.qnetwork_local(states).gather(1,actions)

        # compuet loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # optimize i.e. minimize loss
        self.optimizer.zero_grad()   # zero out gradients
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        
        '''
        soft update of model parameters
        i.e.  target = tau * local + (1-tau) * target

        ---
            local_model: (Pytorch model) weighst copied from
            target_model: (Pytorch model) weights copied to
            tau: for interpolation
        ---
        
        '''

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


class ReplayBuffer():
    '''
    stores experiences for learning
    '''

    def __init__(self, num_actions, buffer_size, batch_size, seed):

        '''
        Initialize an agent

        ---
            num_actions: 
            buffer_size: max buffer size
            batch_size: size of training batches
            seed: random seed
        ---

        '''
        self.num_actions = num_actions
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add_exp(self, state, action, reward, next_state, done):

        '''
        add expereince to memory

        '''

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''
        randomly select batch of experiences
        '''

        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''
        returns current size of memory
        '''

        return len(self.memory)
