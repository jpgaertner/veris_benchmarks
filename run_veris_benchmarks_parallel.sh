#!/bin/bash


timesteps=10
size=10**6
backend=("numpy-mpi")

nprocs_list=(2 4 8)
#nprocs_list=(1 2 4 8 16 32 64 128 256)


# create initial conditions and forcing fields according to grid size
sed -i "s|^problem_size.*|problem_size = $size|" gendata_veris.py
python3 gendata_veris.py

# adjust benchmark according to grid size
sed -i "s|\(problem_size = \).*|\1$size|" benchmarks/veris_dyn_benchmark.py
cp benchmarks/veris_dyn_benchmark.py benchmarks/veris_dyn_${size}_benchmark.py

# request more memory for large grids and many processors;
memory_requ=20
sed -i "s|^#SBATCH --mem.*|#SBATCH --mem=${memory_requ}G|" job.sh

for nprocs in "${nprocs_list[@]}"; do

    # adjust benchmark to selected backend, size and number of processors
    sed -i "s|^python3.*|python3 run_benchmarks.py --only veris_dyn_${size}_benchmark.py -s 5e4 -c $backend -t $timesteps -n $nprocs -o benchmark_results_parallel/${backend}_${nprocs}.json --slurm|" job.sh

    
    sed -i "s|^#SBATCH --ntasks.*|#SBATCH --ntasks=$nprocs|" job.sh

    sbatch job.sh
    echo "job for nprocs=$nprocs submitted"
    
done
