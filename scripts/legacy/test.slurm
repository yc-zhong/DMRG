#!/bin/bash
#SBATCH --job-name=SS
#SBATCH --partition=qnode
#SBATCH --output=out/SS2.out
#SBATCH --error=out/SS2.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
srun --mpi=pmi2 julia -t 1 measure2.jl
