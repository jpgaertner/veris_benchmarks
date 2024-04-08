#!/bin/bash
#SBATCH --account=clidyn.clidyn
#SBATCH --job-name=mitgcm_seaice
#SBATCH --partition=mpp
#SBATCH --qos=12h
#SBATCH -t 1:00:00
#SBATCH --ntasks=128
#SBATCH -o output.txt
#SBATCH -e output.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jan.gaertner@awi.de

# list of hosts that you are running on
hostlist=$(scontrol show hostnames | tr '\n' ',' | rev | cut -c 2- | rev)
echo "hosts: $hostlist" 

# make new files created in this job readable for everybody
umask 022

# load your modules
module load gcc
module load openmpi/4.1.3
module load netcdf-fortran/4.5.4-gcc12.1.0

# maximum possible stacksize
ulimit -s unlimited

# even though we do not run an OpenMP code it is still a good idea to always set this
export OMP_NUM_THREADS=1
cd ${SLURM_SUBMIT_DIR}

srun --cpu_bind=cores ./../build/mitgcmuv
