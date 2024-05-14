#!/bin/bash
#SBATCH --account=clidyn.clidyn
#SBATCH --job-name=veros_seaice
#SBATCH --partition=mpp
##SBATCH --partition=gpu
##SBATCH --gpus=a100:1
##SBATCH --hint=nomultithread
#SBATCH --qos=12h
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=1
#SBATCH -o output.txt
#SBATCH -e output.txt
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jan.gaertner@awi.de

# list of hosts that you are running on
hostlist=$(scontrol show hostnames | tr '\n' ',' | rev | cut -c 2- | rev)
echo "hosts: $hostlist" 

# make new files created in this job readable for everybody
umask 022

# maximum possible stacksize
ulimit -s unlimited

# even though we do not run an OpenMP code it is still a good idea to always set this
export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}

python3 run_benchmarks.py -s 5e4 -c jax-mpi -t 100 --slurm
