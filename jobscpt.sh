#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:05:00
#SBATCH --job-name=laplace
#SBATCH --output=laplace.out
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1

module load nvhpc
TYPE="${TYPE:-float}"
REAL_T_DEF="float"
NVCC_EXTRA=""
case "$TYPE" in
  float) REAL_T_DEF="float" ;;
  double) REAL_T_DEF="double" ;;
  half) REAL_T_DEF="__half"; NVCC_EXTRA="--expt-extended-lambda -DHAS_HALF" ;;
esac
DTYPE_STR="fp32"; [[ "$TYPE" == "double" ]] && DTYPE_STR="fp64"; [[ "$TYPE" == "half" ]] && DTYPE_STR="fp16"

SRC=laplace2d_2.cu
nvcc -O3 -arch=sm_80 -Xptxas=-v $NVCC_EXTRA \
  -DREAL_T=$REAL_T_DEF -DDTYPE_STR="\"$DTYPE_STR\"" \
  $SRC -o laplace || exit 2

# Candidatos (mismos que usabas antes)
# Cobertura amplia (incluye 1,32 y 1,1024, y TODO lo que cumpla bx*by<=1024)
BX_LIST="1,2,4,8,16,32,64,128,256,512,1024"
BY_LIST="1,2,4,8,16,32,64,128,256,512,1024"

TUNE_REPS=${TUNE_REPS:-4}     # iteraciones por candidato dentro del while
CHECK_EVERY=${CHECK_EVERY:-5} # cada cuántas iteraciones calcular el error

srun ./laplace --bx-list="$BX_LIST" --by-list="$BY_LIST" \
               --tune-reps=$TUNE_REPS --check-every=$CHECK_EVERY

