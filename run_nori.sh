#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=36
#SBATCH --time=0-08:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=letr
#SBATCH --output=letr.out

echo "Running LETR with $SLURM_NTASKS tasks, $SLURM_CPUS_PER_TASK cores per task, $SLURM_MEM_PER_CPU memory per core"


python3 create_letr_img.py


echo "Done rendering"

exit 0

