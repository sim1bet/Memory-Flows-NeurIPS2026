#!/bin/bash

#SBATCH --job-name Hopf_conc
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt
#SBATCH --mail-user simone.betteti@ai4i.it
#SBATCH --mail-type ALL
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --account=AIFAC_P02_1086
#SBATCH --partition=boost_usr_prod



srun singularity exec /leonardo/home/userexternal/sbetteti/Hopfield_Concentration/anaconda-wandb.sif bash ./run_all.sh --workers 32 --skip-tex
