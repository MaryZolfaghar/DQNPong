#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 2

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate RLPongDQN

gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n")
echo "gpu" $gpu

python3 train.py \
--use_optim_scheduler \
--initial_lr 0.0003 \
--num_frames 10000000 \
--save_interim_path ../results/DQN/interim/ \
--save_result_path ../results/DQN/results_lr3e_5_10Mframes_scheduler3.npy \
--save_model_path ../results/DQN/model_lr3e_5_10Mframes_scheduler3.pth
