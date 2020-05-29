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
for gpu in $gpus
do
echo "Setting fan for" $gpu "to full"
nvidia_fancontrol full $gpu
done

python3 train.py \
--seed 1 \
--batch_size 32 \
--num_frames 1000000 \
--gamma 0.99 \
--epsilon_start 1.0 \
--epsilon_final 0.01 \
--epsilon_decay 30000 \
--lr 0.0001 \
--optimizer Adam \
--N 1 \
--capacity 100000 \
--save_result_path ../results/DQN/results_default_lr4.npy \
--save_model_path ../results/DQN/model_default_lr4.pth

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
