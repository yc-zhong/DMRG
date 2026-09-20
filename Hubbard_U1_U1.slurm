#!/bin/bash
#SBATCH --job-name=test_F
#SBATCH --partition=qnode
#SBATCH --output=out/test.out
#SBATCH --error=out/test.err
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
export L1=4
export L2=32
export U=8
export Ne=112
export t2=-0.0
export t3=0.0
export model=Square_Hubbard_U1_U1_CBC
export OMP_THREADS=6
export INPUT_BOND_DIMENSION=15000
export BOND_DIMENSION=15000
export STOP_BOND_DIMENSION=20000
srun --mpi=pmi2 julia -t 6 starts.jl
