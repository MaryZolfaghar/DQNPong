"""
Representation of high-dim embedding data
"""

from sklearn.manifold import MDS,TSNE,Isomap,LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import argparse
import time
import numpy as np
import torch
import torch.nn as nn

import gym
from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from models.dqn import QLearner, compute_td_loss, ReplayBuffer


USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser()
# QLearner
parser.add_argument('--batch_size', type=int, default=16,
                    help='')
parser.add_argument('--num_frames', type=int, default=1000000,
                    help='')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Temporal discounting parameter')
parser.add_argument('--N', type=int, default=1,
                    help='Horizon for N-step Q-estimates')
# Wrapper
parser.add_argument('--frame_stack', action='store_true',
                    help='Num of frames to stack, default is using prev four frames')

# Dim reduction representations
parser.add_argument('--embedding_type', choices=['fc_features','out','hidd_features'],
                    default='fc_features',
                    help='what layer output to use as embedding')
parser.add_argument('--dim_red_method', choices=['PCA','KernelPCA','Isomap','LLE','MDS','tSNE'],
                    default='PCA',
                    help='what layer output to use as embedding')
# kernel PCA
parser.add_argument('--kernel_pca', choices=["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"],
                    default='poly',
                    help='what kerel use for PCA')
parser.add_argument('--kpca_gamma', type=float, default=10,
                    help='Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels')
# Saving
parser.add_argument('--save_interim_dim_red', default='../results/DQN/interim/dim_red_results/',
                    help='Path to interim output data file with score history')

def plot_emb_scatter(embedding, labels, title, save_fn):
    colors = ['green','blue','red']

    plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap=ListedColormap(colors))

    plt.title(title)
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    plt.ticklabel_format(style='plain',useOffset=False)
    plt.savefig(save_fn, bbox_inches = 'tight', pad_inches = 0)


#---------------------------------------------------------------------------------------
# load best pre-trainned model
#---------------------------------------------------------------------------------------
args = parser.parse_args()
#------
model_result_path =  '../results/DQN/fmodel_best_19_lr1e-05_frame_1430000_framestack_False_scheduler_True_scheduler2_version2.pth'
replay_initial = 10000 #50000
capacity = 1000000
replay_buffer = ReplayBuffer(capacity)

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env, frame_stack=False)
env = wrap_pytorch(env)

model_new = QLearner(env, args, replay_buffer)
model_new.load_state_dict(torch.load(model_result_path))
model_new = model_new.cuda()

#---------------------------------------------------------------------------------------
# load data that randomely sampled from replay-buffer using best pre-trainned model
#---------------------------------------------------------------------------------------
state      = np.load(save_interim_dim_red + 'state_sample_pretrainned_buffer.npy')
action     = np.load(save_interim_dim_red + 'action_sample_pretrainned_buffer.npy')
reward     = np.load(save_interim_dim_red + 'reward_sample_pretrainned_buffer.npy')
next_state = np.load(save_interim_dim_red + 'next_state_sample_pretrainned_buffer.npy')
done       = np.load(save_interim_dim_red + 'done_sample_pretrainned_buffer.npy')

state_batch      = Variable(torch.FloatTensor(np.float32(state)))
next_state_batch = Variable(torch.FloatTensor(np.float32(next_state)))
action_batch     = Variable(torch.LongTensor(action))
reward_batch     = Variable(torch.FloatTensor(reward))
done_batch       = Variable(torch.FloatTensor(done))


#---------------------------------------------------------------------------------------
# remapping the actions
#---------------------------------------------------------------------------------------
labels = action_batch.cpu().detach().numpy().copy()
actions_remap = labels.copy()

valid_actions = {'stay':0, 'up': 2, 'down': 3}
actions_remap[actions_remap==1] = valid_actions['stay']
actions_remap[actions_remap==4] = valid_actions['up']
actions_remap[actions_remap==5] = valid_actions['down']


#---------------------------------------------------------------------------------------
# extracting the embedding representations
#---------------------------------------------------------------------------------------
embedding_type = args.embedding_type

if embedding_type == 'out':
    state_embedding = model_new.forward(state1)
    state_embeddings = state_embedding.cpu().detach().numpy()
elif embedding_type == 'hidd_features':
    state_embedding = model_new.features(state1)
elif embedding_type == 'fc_features':
    state_batch1 = model_new.features(state_batch)
    state_batch2 = state_batch1.view(state_batch1.size(0), -1)
    fc_output = model_new.fc[0](state_batch2).cpu().detach().numpy()

state_embeddings = fc_output


#---------------------------------------------------------------------------------------
# apply dimentionality reduction method
#---------------------------------------------------------------------------------------
dim_red_method = args.dim_red_method

if dim_red_method=='PCA':
    pca_emb = PCA(n_components=2)
    pca_emb_2d = pca_emb.fit_transform(state_embeddings)
    np.save(args.save_interim_dim_red + 'pca_emb_2d.npy', pca_emb_2d)

elif dim_red_method=='KernelPCA':
    kpca_emb = KernelPCA(n_components=2, kernel=args.kernel_pca, gamma=args.kpca_gamma)
    kpca_emb_2d = kpca_emb.fit_transform(state_embeddings)
    np.save(args.save_interim_dim_red + 'kpca_kernel_%s_gamma%s_emb_2d.npy' \
            %(args.kernel_pca, args.kpca_gamma), kpca_emb_2d)

elif dim_red_method=='Isomap':
    isomap_emb = Isomap(n_components=2)
    isomap_emb_2d = isomap_emb.fit_transform(state_embeddings)
    np.save(args.save_interim_dim_red + 'isomap_emb_2d.npy', isomap_emb_2d)

elif dim_red_method=='LLE':
    lle_emb = LocallyLinearEmbedding(n_components=2)
    lle_emb_2d = lle_emb.fit_transform(state_embeddings)
    np.save(args.save_interim_dim_red + 'lle_emb_2d.npy', lle_emb_2d)

elif dim_red_method=='MDS':
    mds_emb = MDS(n_components=2)
    mds_emb_2d = mds_emb.fit_transform(state_embeddings)
    np.save(args.save_interim_dim_red + 'mds_emb_2d.npy', mds_emb_2d)

elif dim_red_method=='tSNE':
    tsne_emb = TSNE(n_components=2)
    tsne_emb_2d = tsne_emb.fit_transform(state_embeddings)
    np.save(args.save_interim_dim_red + 'tsne_emb_2d.npy', tsne_emb_2d)
