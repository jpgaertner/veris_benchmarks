#!/bin/bash
#SBATCH --account=bk1377
#SBATCH --job-name=mitgcm_seaice
#SBATCH --partition=compute
#SBATCH -t 0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o output.txt
#SBATCH -e output.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jan.gaertner@awi.de
#SBATCH --mem=5G


ulimit -c 0
ulimit -s 1067008 # 1GB
ulimit -l 1067008 # 1GB

export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}

export OMPI_MCA_pml_ucx_opal_mem_hooks=1

srun --cpu_bind=cores ./../build/mitgcmuv
