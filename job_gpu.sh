#!/bin/bash -l
#SBATCH --account=xxx
#SBATCH --job-name=gpu_test
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -t 1:00:00
#SBATCH --output=output.txt
#SBATCH --mem=5G
##SBATCH --exclusive


export OMP_NUM_THREADS=1
python3 run_benchmarks.py --only veris_dyn_10**5_benchmark.py -s 5e4 -n 1 -c jax-gpu -t 10 -o benchmark_results/jax-gpu_10**5.json --slurm
