#!/bin/bash
#SBATCH --account=bk1377
#SBATCH --job-name=jax_gpu_1e7
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -t 0:30:00
#SBATCH -o output_g.txt
#SBATCH -e output_g.txt
#SBATCH --mem=5G

nvidia-smi

ulimit -c 0
ulimit -s 1067008 # 1GB

export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}


python3 run_benchmarks.py -s 5e4 -c jax-gpu -t 100 -n 1 --slurm
