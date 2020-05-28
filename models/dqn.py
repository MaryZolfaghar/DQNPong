"""
Deep Q Network
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()

class QLearner(nn.Module):
    def __init__(self, env, args, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.num_frames = args.num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.N = args.N

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        action = []

        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            ######## YOUR CODE HERE! ########
            # TODO: Given state, you should write code to get the Q value and chosen action
            # Complete the R.H.S. of the following 2 lines and uncomment them
            q_value = self.forward(state)
            action = torch.argmax(q_value)
            ######## YOUR CODE HERE! ########
        else:
            action = random.randrange(self.env.action_space.n)
        return action

def compute_td_loss(model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)), requires_grad=True)
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    ######## YOUR CODE HERE! ########
    # TODO: Implement the Temporal Difference Loss
    q_value = model.forward(state)
    next_q_value = model.forward(next_state)

    next = []
    for ii, dn in enumerate(done):
        if dn:
            next.append(reward)
        else:
            next.append(reward + (gamma ** self.N ) * torch.max(next_q_value))

    next = Variable(torch.FloatTensor(np.float32(next)), requires_grad=True)
    current = [q_value[ii,act] for ii, act in enumerate(action)]
    current = Variable(torch.FloatTensor(np.float32(current)), requires_grad=True)

    loss = torch.sqrt(torch.mean((next - current)**2))
    ######## YOUR CODE HERE! ########
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) #Returns a new deque object initialized left-to-right

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ######## YOUR CODE HERE! ########
        # TODO: Randomly sampling data with specific batch size from the buffer
        # Hint: you may use the python library "random".

        batch = random.sample(self.buffer, batch_size)
        state  = []
        action = []
        reward = []
        next_state = []
        done = []
        for sample in batch:
            state.append(sample[0])
            action.append(sample[1])
            reward.append(sample[2])
            next_state.append(sample[3])
            done.append(sample[4])

        # If you are not familiar with the "deque" python library, please google it.
        ######## YOUR CODE HERE! ########
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
