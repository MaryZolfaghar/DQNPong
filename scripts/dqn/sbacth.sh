#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH -c 2

# slurm-90333.out sbatch scripts/dqn/dqn_pong_deafults_lr1e_2.sh #not good; Last-10:-20.3  Best:-19.7
# slurm-90334.out sbatch scripts/dqn/dqn_pong_deafults_lr1e_3.sh #not good; Last-10:-20.8  Best:-19.9
# slurm-90335.out sbatch scripts/dqn/dqn_pong_deafults_lr1e_4.sh # Last-10:-14.8  Best:-13.5
# slurm-90331.out sbatch scripts/dqn/dqn_pong_deafults_lr2e_4.sh #not good, Last-10:-18.8 Best:-15.7
# slurm-90336.out sbatch scripts/dqn/dqn_pong_deafults_lr3e_5.sh # Last-10:-13.7  Best:-12.6
# slurm-90337.out sbatch scripts/dqn/dqn_pong_deafults_lr5e_4.sh #not good; Last-10:-20.4  Best:-20.1
# slurm-90338.out sbatch scripts/dqn/dqn_pong_deafults_lr6e_5.sh # Last-10:-16.0  Best:-14.0
#
# # best scripts
# slurm-90335.out sbatch scripts/dqn/dqn_pong_deafults_lr1e_4.sh # Last-10:-14.8  Best:-13.5
# slurm-90336.out sbatch scripts/dqn/dqn_pong_deafults_lr3e_5.sh # Last-10:-13.7  Best:-12.6
# slurm-90338.out sbatch scripts/dqn/dqn_pong_deafults_lr6e_5.sh # Last-10:-16.0  Best:-14.0

#new scripts
# sbatch scripts/dqn/dqn_pong_deafults_lr1e_5.sh # Last-10:  Best:
# sbatch scripts/dqn/dqn_pong_deafults_lr2e_5.sh # Last-10:  Best:
# sbatch scripts/dqn/dqn_pong_deafults_lr1e_6.sh # Last-10:  Best:
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_6.sh # Last-10:  Best:
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_diffKernel.sh
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes.sh
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_4prevframes.sh
sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_scheduler.sh
