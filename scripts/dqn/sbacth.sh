#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH -c 2

# sbatch scripts/dqn/dqn_pong_deafults_lr1e_2.sh
# sbatch scripts/dqn/dqn_pong_deafults_lr1e_3.sh
# sbatch scripts/dqn/dqn_pong_deafults_lr1e_4.sh
sbatch scripts/dqn/dqn_pong_deafults_lr2e_4.sh
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5.sh
# sbatch scripts/dqn/dqn_pong_deafults_lr5e_4.sh
# sbatch scripts/dqn/dqn_pong_deafults_lr6e_5.sh
