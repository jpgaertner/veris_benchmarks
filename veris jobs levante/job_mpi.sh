#!/bin/bash
#SBATCH --account=bk1377
#SBATCH --job-name=np_mpi_1e6
#SBATCH --partition=compute
#SBATCH -t 1:00:00
#SBATCH --mem=20G
#SBATCH -o output_jm.txt
#SBATCH -e output_jm.txt


ulimit -c 0
ulimit -s 1067008 # 1GB

export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}


python3 run_benchmarks.py -s 5e4 -c jax-mpi -t 100 -n 128 --slurm
