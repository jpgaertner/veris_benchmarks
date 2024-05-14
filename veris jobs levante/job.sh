#!/bin/bash
#SBATCH --account=bk1377
#SBATCH --job-name=jax_1e7
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 0:30:00
#SBATCH --mem=2G
#SBATCH -o output.txt
#SBATCH -e output.txt


ulimit -c 0
ulimit -s 1067008 # 1GB

export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}


python3 run_benchmarks.py -s 5e4 -c jax -t 100 -n 1 --slurm
