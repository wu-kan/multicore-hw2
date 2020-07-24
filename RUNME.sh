#!/bin/bash
#SBATCH -J WuK_scaffold
#SBATCH -p gpu_v100
#SBATCH -N 1
#SBATCH --exclusive

#module load cmake/3.14.3-gcc-4.8.5
#module load CUDA/10.1.2

mkdir -p sources/build
cd sources/build
rm -fr *

cmake ..
make

cd ../..
sources/build/main