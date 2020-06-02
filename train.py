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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from models.dqn import QLearner, compute_td_loss, ReplayBuffer

USE_CUDA = torch.cuda.is_available()


parser = argparse.ArgumentParser()
# CUDA
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')
# Wrapper
parser.add_argument('--frame_stack', action='store_true',
                    help='Num of frames to stack, default is using prev four frames')




# QLearner
parser.add_argument('--batch_size', type=int, default=16,
                    help='')
parser.add_argument('--num_frames', type=int, default=1000000,
                    help='')
# Environment
parser.add_argument('--render', type=int, default=0,
                    help='Rendering the environment state')
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
parser.add_argument('--use_optim_scheduler', action='store_true',
                    help='Whether use scheduler for the optimizer or not')
parser.add_argument('--initial_lr', type=float, default=0.0003,
                    help='Initial Learning rate of optimizer')
parser.add_argument('--step_size', type=int, default=50000,
                    help='Size of step for the optimizer scheduler')
# ReplayBuffer
parser.add_argument('--capacity', type=int, default=1000000,
                    help='Number of states to store in the replay buffer')
# Saving Results
parser.add_argument('--save_result_path', default='../results/DQN/results.npy',
                    help='Path to output data file with score history')
parser.add_argument('--save_model_path', default='../results/DQN/weights_only.pth',
                    help='Path to output data file for saving the trainned model')
parser.add_argument('--save_interim_path', default='../results/DQN/interim/',
                    help='Path to interim output data file with score history')
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
    env = wrap_deepmind(env, args.frame_stack)
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
        if args.use_optim_scheduler:
            optimizer = optim.Adam(model_Q.parameters(), lr=args.initial_lr)
            # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
        else:
            optimizer = optim.Adam(model_Q .parameters(), args.lr)

    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model_Q.parameters(), args.lr)

    if USE_CUDA:
        model_Q = model_Q.cuda()
        model_target_Q = model_target_Q.cuda()

    # Training loop
    epsilon_by_frame = lambda frame_idx: args.epsilon_final + (args.epsilon_start - args.epsilon_final) * math.exp(-1. * frame_idx / args.epsilon_decay)

    losses = []
    learning_rates = []
    all_rewards = []
    episode_reward = 0
    num_param_updates = 0
    mean_reward = -float('nan')
    mean_reward2 = -float('nan')
    best_mean_reward = -float('inf')
    best_mean_reward2 = -float('inf')
    time_history = [] # records time (in sec) of each episode
    old_lr = args.initial_lr
    state = env.reset()
    start_time_frame = time.time()
    for frame_idx in range(1, args.num_frames + 1):
        start_time = time.time()

        epsilon = epsilon_by_frame(frame_idx)
        action = model_Q.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            time_history.append(time.time() - start_time)
            episode_reward = 0

        if args.render==1:
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
            if num_param_updates % args.target_update_freq == 0:
                model_target_Q.load_state_dict(model_Q.state_dict())

            if args.use_optim_scheduler:
                scheduler.step(mean_reward2)
                new_lr = scheduler.get_last_lr()
                if new_lr != old_lr:
                    learning_rates.append(new_lr)
                    print('NewLearningRate: ', new_lr)
                old_lr = new_lr

        if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
            print("Preparing replay buffer with len -- ", len(replay_buffer),
                  "Frame:", frame_idx,
                  "Total time so far:", (time.time() - start_time_frame))

        if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
            mean_reward = np.mean(all_rewards[-10:])
            mean_reward2 = np.mean(all_rewards[-100:])
            best_mean_reward = max(best_mean_reward, mean_reward)
            best_mean_reward2 = max(best_mean_reward2, mean_reward2)
            print("Frame:", frame_idx,
                  "Loss:", np.mean(losses),
                  "Total Rewards:", all_rewards[-1],
                  "Average Rewards over all frames:", np.mean(all_rewards),
                  "Last-10 average reward:", mean_reward,
                  "Best mean reward of last-10:", best_mean_reward,
                  "Last-100 average reward:", mean_reward2,
                  "Best mean reward of last-100:", best_mean_reward2,
                  "Time:", time_history[-1],
                  "Total time so far:", (time.time() - start_time_frame))

        if frame_idx % args.save_freq_frame == 0:
            results = [losses, all_rewards, time_history]
            torch.save(model_Q.state_dict(), args.save_model_path)
            np.save(args.save_result_path, results)
        if frame_idx == 10000:
            results = [losses, all_rewards, time_history]
            torch.save(model_Q.state_dict(), args.save_interim_path + \
                      'model_lr%s_frame_%s_framestack_%s_scheduler_%s_RPlateau.pth'\
                       %(args.lr,frame_idx, args.frame_stack, args.use_optim_scheduler))
            np.save(args.save_interim_path + \
                   'results_lr%s_frame_%s_framestack_%s_scheduler_%s_RPlateau.npy' \
                    %(args.lr, frame_idx, args.frame_stack, args.use_optim_scheduler), \
                    results)

        if frame_idx % 500000 == 0:
            results = [losses, all_rewards, time_history]
            torch.save(model_Q.state_dict(), args.save_interim_path + \
                      'model_lr%s_frame_%s_framestack_%s_scheduler_%s_RPlateau.pth' \
                      %(args.lr,frame_idx, args.frame_stack, args.use_optim_scheduler))
            np.save(args.save_interim_path + \
                   'results_lr%s_frame_%s_framestack_%s_scheduler_%s_RPlateau.npy' \
                   %(args.lr,frame_idx, args.frame_stack, args.use_optim_scheduler), \
                    results)

            # model_new = NeuralNet()
            # model_new.load_state_dict(torch.load('weights_only.pth'))

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
