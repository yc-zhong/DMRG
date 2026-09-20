#!/bin/bash
#SBATCH --job-name=test_F
#SBATCH --partition=cnode
#SBATCH --output=out/test1.out
#SBATCH --error=out/test1.err
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=8
export L1=4
export L2=4
export U=8
export Ne=14
export t2=0
export t3=0
export Spectrum=1
export OMP_THREADS=8
export model=Square_Hubbard_U1_SU2_CBC
export INPUT_BOND_DIMENSION=100
export BOND_DIMENSION=100
export STOP_BOND_DIMENSION=100
srun --mpi=pmi2 julia -t 12 start.jl
