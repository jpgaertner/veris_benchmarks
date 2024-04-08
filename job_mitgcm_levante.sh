#!/bin/bash
#SBATCH --account=bk1377
#SBATCH --job-name=mitgcm_seaice
#SBATCH --partition=compute
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH -o output.txt
#SBATCH -e output.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jan.gaertner@awi.de


ulimit -s unlimited

export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}

export OMPI_MCA_pml_ucx_opal_mem_hooks=1

srun --cpu_bind=cores ./../build/mitgcmuv