"""
"""

import argparse
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim

from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from models.dqn import QLearner, compute_td_loss, ReplayBuffer

parser = argparse.ArgumentParser()
# CUDA
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')
# QLearner
parser.add_argument('--batch_size', type=int, default=16,
                    help='')
parser.add_argument('--num_frames', type=int, default=1000000,
                    help='')
# Training
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Temporal discounting parameter')
parser.add_argument('--epsilon_start', type=float, default=1.0,
                    help='Initial probability of selecting random action')
parser.add_argument('--epsilon_final', type=float, default=0.01,
                    help='Final probability of selecting random action')
parser.add_argument('--epsilon_decay', type=float, default=30000,
                    help='Decay for probability of selecting random action') # epsilon_decay = 0.99
parser.add_argument('--N', type=int, default=1,
                    help='Horizon for N-step Q-estimates')
parser.add_argument('--number_of_updates', type=int, default=10,
                    help='Number of updates for each batch with batch_size')
parser.add_argument('--target_update_freq', type=int, default=10000,
                    help='Copy current model to target model')
# Optimization
parser.add_argument('--optimizer', choices=['Adam','RMSprop'],
                    default='Adam',
                    help='Optimizer to use for training')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='Learning rate of optimizer (default from mjacar)')
# ReplayBuffer
parser.add_argument('--capacity', type=int, default=1000000,
                    help='Number of states to store in the replay buffer')
# Saving Results
parser.add_argument('--save_result_path', default='../results/DQN/results.npy',
                    help='Path to output data file with score history')
parser.add_argument('--save_model_path', default='../results/DQN/weights_only.pth',
                    help='Path to output data file for saving the trainned model')
parser.add_argument('--save_freq_frame', type=int, default=100000,
                    help='Save model and results every save_freq_frame times')


def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using cuda: ", use_cuda)

    # Environment
    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    # Random seed
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initializing
    replay_initial = 10000 #50000
    replay_buffer = ReplayBuffer(args.capacity)
    # model = QLearner(env, args, replay_buffer)
    # Initialize target q function and q function
    model_Q = QLearner(env, args, replay_buffer)
    model_target_Q = QLearner(env, args, replay_buffer)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model_Q .parameters(), args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model_Q.parameters(), args.lr)

    if USE_CUDA:
        model_Q = model_Q.cuda()
        model_target_Q = model_target_Q.cuda()

    # Training loop
    epsilon_by_frame = lambda frame_idx: args.epsilon_final + (args.epsilon_start - args.epsilon_final) * math.exp(-1. * frame_idx / args.epsilon_decay)

    losses = []
    all_rewards = []
    episode_reward = 0
    num_param_updates = 0
    mean_reward = -float('nan')
    best_mean_reward = -float('inf')
    time_history = [] # records time (in sec) of each episode

    state = env.reset()
    start_time_frame = time.time()
    for frame_idx in range(1, args.num_frames + 1):
        start_time = time.time()

        epsilon = epsilon_by_frame(frame_idx)
        action = model_Q.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        print('Frame', frame_idx, \
              'state shape', state.shape, \
              'action', action, \
              'next_state shape', next_state.shape, \
              'reward', reward, \
              'done', done, '\n')
              
        state = next_state
        episode_reward += reward
        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            time_history.append(time.time() - start_time)
            episode_reward = 0

        if render:
            env.render()

        if len(replay_buffer) > replay_initial:
            for nou in range(args.number_of_updates):
                loss = compute_td_loss(model_Q, model_target_Q, args.batch_size, args.gamma, replay_buffer, args.N)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.data.cpu().numpy())
                num_param_updates += 1
            # Periodically update the target network by Q network to target Q network
            if num_param_updates % target_update_freq == 0:
                model_target_Q.load_state_dict(model_Q.state_dict())

        if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
            print("Preparing replay buffer with len -- ", len(replay_buffer),
                  "Frame:", frame_idx,
                  "Total time so far:", (time.time() - start_time_frame))
            # print('#Frame: %d, preparing replay buffer' % frame_idx)

        if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
            mean_reward = np.mean(all_rewards[-10:])
            best_mean_reward = max(best_mean_reward, mean_reward)
            print("Frame:", frame_idx,
                  "Loss:", np.mean(losses),
                  "Total Rewards:", all_rewards[-1],
                  "Average Rewards over all frames:", np.mean(all_rewards),
                  "Last-10 average reward:", mean_reward,
                  "Best mean reward of last-10:", best_mean_reward,
                  "Time:", time_history[-1],
                  "Total time so far:", (time.time() - start_time_frame))
            # print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses)))
            # print('Last-10 average reward: %f' % np.mean(all_rewards[-10:]))

        if frame_idx % save_freq_frame == 0:
            results = [losses, all_rewards, time_history]
            torch.save(model_Q.state_dict(), args.save_model_path)
            np.save(args.save_result_path, results)
            # model_new = NeuralNet()
            # model_new.load_state_dict(torch.load('weights_only.pth'))

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
