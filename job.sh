#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:05:00
#SBATCH --job-name=laplace_job
#SBATCH --output=laplace_out_gpu_cu.txt
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:2


echo "Nodo: $SLURM_NODELIST"
echo "GPUs visibles: $CUDA_VISIBLE_DEVICES"
nvidia-smi 
# nvidia-smi

# 1. Cargar módulos
module load nvhpc
# module load craype-accel-nvidia80


# 1 CPU NO openMp , no optimization
# g++ -O0  laplace2d.cpp -o laplace

#  2 CPU openMp , it uses GPU due the pragma
# g++ -O3 -fopenmp laplace2d.cpp -o laplace 

# 3. COMPILAR GPU
# nvc++ -mp=gpu -gpu=cc70 -Ofast laplace2d.cpp -o laplace -Minfo=accel,mp

# 4 cu file
nvcc -O3 -arch=sm_70 laplace2d.cu -o laplace

# nvcc -O3 -arch=sm_70 -Xptxas -v -lineinfo -Xptxas -dlcm=ca laplace2d.cu -o laplace


#vect info 
# g++ -O3 -march=native -fopt-info-vec-optimized -fopt-info-vec-missed laplace2d.cpp -fopenmp




# nvc++ -O3 -mp -Minfo=mp laplace2d.cpp -o laplace


# 3. Ejecutar con srun
srun ./laplace
