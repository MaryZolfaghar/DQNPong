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
--frame_stack \
--lr 0.00003 \
--num_frames 10000000 \
--save_interim_path ../results/DQN/interim/ \
--interim_fn 4prevframes_version2 \
--save_result_path ../results/DQN/results_lr3e_5_10Mframes_4prevframes_version2.npy \
--save_model_path ../results/DQN/model_lr3e_5_10Mframes_4prevframes_version2.pth
