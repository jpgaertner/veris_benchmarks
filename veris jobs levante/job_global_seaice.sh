#!/bin/bash -l
#SBATCH -p compute
#SBATCH -A bk1377
#SBATCH --job-name=veros
#SBATCH -t 0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=30G
#SBATCH -o output.txt
#SBATCH -e output.txt


export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}


srun --mpi=pmi2 --cpu-bind=core veros run seaice_global_4deg.py -b numpy -n 2 2
