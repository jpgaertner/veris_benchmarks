#!/bin/bash -l
#SBATCH --account=xxx
#SBATCH --job-name=image
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH -t 0:10:00
#SBATCH --output=output.txt
#SBATCH --mem=5G
#SBATCH --exclusive


python3 run_benchmarks.py --only veris_img_dyn_benchmark.py -s 5e4 -n 1 -c jax-gpu -t 1440 -o dump.json
