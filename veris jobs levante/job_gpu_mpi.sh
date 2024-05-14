#!/bin/bash
#SBATCH --account=bk1377
#SBATCH --job-name=jax_gpu_mpi_1e7_n4
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH -t 0:30:00
#SBATCH -o output_g4.txt
#SBATCH -e output_g4.txt
#SBATCH --mem=20G

nvidia-smi

ulimit -c 0
ulimit -s 1067008 # 1GB

export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}


python3 run_benchmarks.py -s 5e4 -c jax-gpu-mpi -t 100 -n 4 --slurm
