#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH -c 2


sbatch scripts/dqn/dqn_pong_deafults_lr2.sh
sbatch scripts/dqn/dqn_pong_deafults_lr4.sh
sbatch scripts/dqn/qn_pong_RMSprop_bacth8_lr4.sh
sbatch scripts/dqn/dqn_pong_RMSprop_lr4.sh
