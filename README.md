# veris_benchmarks
Benchmark and simulation routines, results, and plotting scripts for the publication

Gärtner, J. P., Losch, M., Jochum, M., and Nuterman, R. (2025). Veris: Fast &amp; Efficient Sea-Ice Modeling in Python with GPU Acceleration.


Setting up Veros & Veris:

Download Veros via git clone https://github.com/team-ocean/veros.git
Download Veris via git clone https://github.com/team-ocean/veris.git

Create the Veros conda environment via
conda env create -f conda-environment.yml
conda activate veros
conda install python=3.10
pip install veros
pip install veris
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
conda install h5py=3.11.0=mpi_mpich_py310h086384f_0
conda install -c conda-forge mpi4jax


Running the Veris benchmarks:

Add the files veris_dyn_benchmark.py, veris_img_dyn_benchmark.py and model.py to veros/benchmarks/

Adjust the path for the forcing fields in veris_dyn_benchmark.py and veris_img_dyn_benchmark.py

Add the files gendata_veris.py, run_veris_benchmarks.sh, run_veris_benchmarks_parallel.sh, job.sh, job_gpu.sh, plot_benchmark_results.ipynb, create_forcing_img.ipynb, read_veris_output.ipynb, job_image.sh to veros/

Adjust the account name in the job scripts

Create the folders benchmark_results, benchmark_results_parallel in veros/

Adjust and run the files run_veris_benchmarks.sh & run_veris_benchmarks_parallel.sh

For plotting the results of the single core benchmarks, run plot_benchmark_results.ipynb


Creating Fig.1:

Run create_forcing_img.ipynb (adjust the grid size if you want)

Sbatch job_image.sh

Run read_veris_output.ipynb
