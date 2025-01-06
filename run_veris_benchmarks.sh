#!/bin/bash


#backends=("numpy" "jax" "jax-gpu")
#sizes=(10**3 10**4 10**5 10**6 10**7)
timesteps=10

backends=("jax")
sizes=(10**3 10**4 10**5)

for size in "${sizes[@]}"; do

    # create initial conditions and forcing fields according to grid size
    sed -i "s|\(problem_size = \).*|\1$size|" gendata_veris.py
    python3 gendata_veris.py

    # adjust benchmark according to grid size
    sed -i "s|\(problem_size = \).*|\1$size|" benchmarks/veris_dyn_benchmark.py
    cp benchmarks/veris_dyn_benchmark.py benchmarks/veris_dyn_${size}_benchmark.py
    
    # request more memory for large grids
    if [[ "$size" == 10**7 ]]; then
        memory_requ=50
    else
        memory_requ=5
    fi

    for backend in "${backends[@]}"; do

        # select job script
        if [[ "$backend" != "jax-gpu" ]]; then
            job_script="job.sh"
        else
            job_script="job_gpu.sh"
        fi
        
        # adjust job script to selected backend
        sed -i "s|^python3.*|python3 run_benchmarks.py --only veris_dyn_${size}_benchmark.py -s 5e4 -n 1 -c $backend -t $timesteps -o benchmark_results/${backend}_${size}.json --slurm|" "$job_script"
        sed -i "s|^\(#SBATCH --mem=\).*|\1${memory_requ}G|" "$job_script"
        sbatch $job_script
        
    done
done
