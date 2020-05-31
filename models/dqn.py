"""
Deep Q Network
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), \
                               requires_grad=True)
            ######## YOUR CODE HERE! ########
            # TODO: Given state, you should write code to get the Q value and chosen action
            # Complete the R.H.S. of the following 2 lines and uncomment them
            q_value = self.forward(state)
            action = torch.argmax(q_value).item()
            ######## YOUR CODE HERE! ########
        else:
            action = random.randrange(self.env.action_space.n)
        return action

def compute_td_loss(model_Q, model_target_Q, batch_size, gamma, replay_buffer, N):

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    ######## YOUR CODE HERE! ########
    # TODO: Implement the Temporal Difference Loss

    # Compute current Q value, q_func takes only state and output value for every state-action pair
    # We choose Q based on action taken.
    q_value = model_Q.forward(state)
    current_q_value = q_value.gather(1, action.view(-1,1))
    # Compute next Q value based on which action gives max Q values
    # Detach variable from the current graph since we don't want gradients for next Q to propagated
    with torch.no_grad():
        next_q_value = model_target_Q.forward(next_state).detach()
        target_q_value = reward + (1-done) * gamma * torch.max(next_q_value.detach(),dim=1)[0]
        target_q_value = target_q_value.view(-1,1)
    # current = Variable(torch.FloatTensor(np.float32(current)))
    # next = Variable(torch.FloatTensor(np.float32(next)), requires_grad=True)

    #
    # print('In loss function', \
    #       'current_q_value shape', current_q_value.shape, \
    #       'action shape', action.shape, \
    #       'target_q_val shape', target_q_value.shape, \
    #       'reward', reward.shape, \
    #       'done', done.shape, '\n')


    # loss = torch.mean((target_q_val - current_q_value)**2))
    loss = F.smooth_l1_loss(current_q_value, target_q_value)
    ######## YOUR CODE HERE! ########
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        #Returns a new deque object initialized left-to-right
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ######## YOUR CODE HERE! ########
        # TODO: Randomly sampling data with specific batch size from the buffer
        # Hint: you may use the python library "random".

        batch = random.sample(self.buffer, batch_size)
        state,action,reward,next_state,done = zip(*batch)
        # state  = []
        # action = []
        # reward = []
        # next_state = []
        # done = []
        # for sample in batch:
        #     state.append(sample[0])
        #     action.append(sample[1])
        #     reward.append(sample[2])
        #     next_state.append(sample[3])
        #     done.append(sample[4])

        # If you are not familiar with the "deque" python library, please google it.
        ######## YOUR CODE HERE! ########
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)
