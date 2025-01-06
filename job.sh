#!/bin/bash -l
#SBATCH --account=xxx
#SBATCH --job-name=veris_bm
#SBATCH --partition=compute
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=output.txt
#SBATCH --mem=5G
#SBATCH --exclusive


python3 run_benchmarks.py --only veris_dyn_10**5_benchmark.py -s 5e4 -n 1 -c jax -t 10 -o benchmark_results/jax_10**5.json --slurm
