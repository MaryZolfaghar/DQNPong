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


#best one so far : cat slurm-90398.out
# Namespace(N=1, batch_size=16, capacity=1000000, epsilon_decay=30000, epsilon_final=0.01, epsilon_start=1.0, frame_stack=True, gamma=0.99, lr=3e-05, num_frames=10000000, number_of_updates=10, optimizer='Adam', render=0, save_freq_frame=100000, save_interim_path='../results/DQN/interim/', save_model_path='../results/DQN/model_lr3e_5_10Mframes_4prevframes.pth', save_result_path='../results/DQN/results_lr3e_5_10Mframes_4prevframes.npy', seed=1, target_update_freq=10000)



# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_diffKernel.sh
# # sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes.sh
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_4prevframes.sh # slurm-90566.out      slurm-90564.out
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_scheduler.sh # job 90443 with warning - job 90447 without warning
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_scheduler2.sh # job 90446
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_scheduler3.sh # job
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_deafultSetting.sh
#
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_4prevframes_RMSopt.sh # slurm-90567.out
#
# sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_4prevframes_scheduler2.sh #  slurm-90568.out



#model_lr3e_5_10Mframes_4prevframes_version2.pth        slurm-90564.out
#model_lr3e_5_10Mframes_4prevframes_scheduler2.pth      slurm-90568.out
#model_lr3e_5_10Mframes_scheduler2.pth                  slurm-90446.out
#model_lr3e_5_10Mframes_4prevframes.pth                 slurm-90398.out


# Final Running
sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_4prevframes_scheduler2.sh  #90609 -> slow and not good that much - Frame: 1990k Loss: 0.004627831 Last-10 average reward: 11.4 Best mean reward of last-10: 16.8
sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_4prevframes.sh #90610 -> start to go down ->                       Frame: 1740k Loss: 0.004173764 Last-10 average reward: 19.1 Best mean reward of last-10: 19.1 Last-100 average reward: 16.94 Best mean reward of last-100: 18.23
sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_scheduler2_version2.sh #90611 -> sound promising                   Frame: 1430k Loss: 0.004129095 Last-10 average reward: 19.6 Best mean reward of last-10: 19.6 Last-100 average reward: 18.02 Best mean reward of last-100: 18.02
sbatch scripts/dqn/dqn_pong_deafults_lr3e_5_10Mframes_4prevframes_RMSopt.sh #90612 -> starts to go down                  Frame: 1770k Loss: 0.0050163595 Last-10 average reward: 18.7 Best mean reward of last-10: 18.7 Last-100 average reward: 15.79 Best mean reward of last-100: 15.79

fmodel_best_19_lr1e-05_frame_1430000_framestack_False_scheduler_True_scheduler2_version2.pth
fmodel_best_19_lr3e-05_frame_1740000_framestack_True_scheduler_False_4prevframes_version2.pth
model_best_19_lr1e-05_frame_1560000_framestack_True_scheduler_True.pth


running
90568  -> sounds promising , fluctuates a lot and is going down but might do better accoridng to its trend - model_lr3e_5_10Mframes_4prevframes_scheduler2.pth
          Frame: 1560k Loss: 0.0040654526 Last-10 average reward: 19.4 Best mean reward of last-10: 19.4 Last-100 average reward: 16.34 Best mean reward of last-100: 16.97 Time: 0.021749496459960938 Total time so far: 124916.45731282234

90564   -> doesnt have any log! scacnel

90612  - above - Frame: 1770k Best mean reward of last-10: 18.7

NewLearningRate:  [2.9403e-05]
model_lr3e_5_10Mframes_scheduler2_version2.pth
"BEST" 90611  - above - Frame: 1430k Best mean reward of last-10: 19.6

90610  - above - Frame: 2100k Best mean reward of last-10: 19.4
90609  - above - Frame: 2450k Best mean reward of last-10: 18.7

90446 -> scancelled , model_lr3e_5_10Mframes_scheduler2.pth was good but starts going down! check it is has been save around 1M frames
         Frame: 1390k Loss: 0.0031094998 Last-10 average reward: 17.3 Best mean reward of last-10: 19.7 Last-100 average reward: 16.92 Best mean reward of last-100: 18.14 Time: 0.01653909683227539 Total time so far: 103813.75174665451
         Frame: 1400k Loss: 0.0030980392 Last-10 average reward: 19.9 Best mean reward of last-10: 19.9 Last-100 average reward: 17.11 Best mean reward of last-100: 18.14 Time: 0.017970561981201172 Total time so far: 104652.40115499496
         path: ../results/DQN/model_lr3e_5_10Mframes_scheduler2.pth
         path: ../results/DQN/interim/cannot find!
