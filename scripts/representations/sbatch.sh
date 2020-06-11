#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH -c 2


# Final Running
sbatch scripts/representations/dim_red_reps_isomap.sh
sbatch scripts/representations/dim_red_reps_lle.sh
sbatch scripts/representations/dim_red_reps_mds.sh
sbatch scripts/representations/dim_red_reps_pca.sh
sbatch scripts/representations/dim_red_reps_tSNE.sh
sbatch scripts/representations/dim_red_reps_kpca_cosine_gamma10.sh
sbatch scripts/representations/dim_red_reps_kpca_poly_gamma10.sh
sbatch scripts/representations/dim_red_reps_kpca_rbf_gamma10.sh
sbatch scripts/representations/dim_red_reps_kpca_rbf_gamma5.sh
sbatch scripts/representations/dim_red_reps_kpca_sigmoid_gamma10.sh
