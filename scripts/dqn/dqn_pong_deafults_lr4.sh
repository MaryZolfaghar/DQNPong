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
conda activate DeepLearningEEG

gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n")
for gpu in $gpus
do
echo "Setting fan for" $gpu "to full"
nvidia_fancontrol full $gpu
done

python train.py \
--use_cuda \
--seed 1 \
--batch_size 32 \
--num_frames 1000000 \
--gamma 0.99 \
--epsilon_start 1.0 \
--epsilon_final 0.01 \
--epsilon_decay 0.99 \
--lr 1e-4 \
--optimizer Adam \
--N 100 \
--capacity 100000 \
--save_result_path ../results/DQN/results_default_lr4.npy

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
