#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:05:00
#SBATCH --job-name=laplace_tune
#SBATCH --output=laplace_tune.out
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1

echo "Node: $SLURM_NODELIST"
nvidia-smi
module load nvhpc

TYPE="${TYPE:-float}"
REAL_T_DEF="float"
NVCC_EXTRA=""

case "$TYPE" in
  float) REAL_T_DEF="float" ;;
  double) REAL_T_DEF="double" ;;
  half) REAL_T_DEF="__half"; NVCC_EXTRA="--expt-extended-lambda -DHAS_HALF" ;;
esac

DTYPE_STR="fp32"
[[ "$TYPE" == "double" ]] && DTYPE_STR="fp64"
[[ "$TYPE" == "half" ]] && DTYPE_STR="fp16"

SRC=laplace2d_2.cu

# 1) Compila tuner
nvcc -O3 -arch=sm_70 -Xptxas=-v $NVCC_EXTRA \
  -DREAL_T=$REAL_T_DEF -DDTYPE_STR="\"$DTYPE_STR\"" -DUSE_FIXED_BLOCK=0 \
  $SRC -o laplace_tuner || exit 2

# 2) Barrido interno
BX_LIST=${BX_LIST:-"32,64,128,256,512,1024"}
BY_LIST=${BY_LIST:-"1,2,4,8,16,32"}
: > jacobi_log.txt
: > tune.csv

BEST_LINE=$( srun ./laplace_tuner --bx-list="$BX_LIST" --by-list="$BY_LIST" | tee tune_run.txt | awk '/GLOBAL_BEST/{for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="bx")bx=a[2];if(a[1]=="by")by=a[2];if(a[1]=="gflops")gf=a[2];}print bx,by,gf}' | tail -n1 )

BEST_BX=$(echo "$BEST_LINE" | awk '{print $1}')
BEST_BY=$(echo "$BEST_LINE" | awk '{print $2}')
BEST_GF=$(echo "$BEST_LINE" | awk '{print $3}')
echo "Best block found: bx=$BEST_BX by=$BEST_BY GFLOPS=$BEST_GF" | tee -a tune_run.txt

# 3) Compila binario fijo
OUT_BIN=laplace_opt_bx${BEST_BX}_by${BEST_BY}
nvcc -O3 -arch=sm_80 -Xptxas=-v $NVCC_EXTRA \
  -DREAL_T=$REAL_T_DEF -DDTYPE_STR="\"$DTYPE_STR\"" \
  -DUSE_FIXED_BLOCK=1 -DBX_DEF=$BEST_BX -DBY_DEF=$BEST_BY \
  $SRC -o "$OUT_BIN" || exit 3

echo "Binary with optimal block saved as: $OUT_BIN" | tee -a tune_run.txt
echo "All logs: jacobi_log.txt / tune_run.txt"
