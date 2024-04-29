#!/bin/bash
#SBATCH --account=bk1377
#SBATCH --job-name=veris_seaice
##SBATCH --partition=compute
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=10G
#SBATCH -o output.txt
#SBATCH -e output.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jan.gaertner@awi.de

# the -n is necessary when using mpi, but breaks the run when not mpi!!!
# submitting works on both ssh and via jupyterlab gpu

ulimit -s unlimited

export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}

#export OMPI_MCA_pml_ucx_opal_mem_hooks=1

python3 run_benchmarks.py -s 5e4 -c jax-gpu-mpi -t 10 -n 2 --slurm