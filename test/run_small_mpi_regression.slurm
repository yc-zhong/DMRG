#!/usr/bin/env bash
#SBATCH --partition=amd_test
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_d1000_20260917"
readonly source_root="${migration_root}/small_mpi_regression/source_d16"
readonly run_root="${migration_root}/small_mpi_regression/run_${SLURM_JOB_ID}"
readonly serial_output="${run_root}/serial"
readonly mpi_output="${run_root}/mpi_r8"

test -d "${source_root}"
test ! -e "${run_root}"
mkdir -p "${run_root}"
cd "${engine_root}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1
export OMP_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export model=Square_Hubbard_U1_U1_OBC_LeftEdge
export Lx=3 L2=2 Ne=6 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=16 INPUT_BOND_DIMENSION=16 STOP_BOND_DIMENSION=16
export START_MPS_DIR="${source_root}"
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SWEEP_DIMS=16

export OUTPUT_MPS_DIR="${serial_output}"
export RUN_TAG="small_serial_${SLURM_JOB_ID}"
export SVD_MODE=serial
/usr/bin/time -v julia --project=. start.jl

export OUTPUT_MPS_DIR="${mpi_output}"
export RUN_TAG="small_mpi_r8_${SLURM_JOB_ID}"
export SVD_MODE=mpi
/usr/bin/time -v julia --project=. -e '
    using MPI
    nranks=parse(Int,ENV["SLURM_NTASKS"])
    launcher=MPI.mpiexec()
    child=`julia --project=. start.jl`
    run(`$launcher -n $nranks $child`)
'

serial_current="${serial_output}/$(tr -d '\n' < "${serial_output}/CURRENT")"
mpi_current="${mpi_output}/$(tr -d '\n' < "${mpi_output}/CURRENT")"
cd "${migration_root}/converter"
julia --project=. compare_tensorkit_mps.jl "${serial_current}" "${mpi_current}"

printf 'source=%s\nserial=%s\nmpi=%s\n' \
  "${source_root}" "${serial_current}" "${mpi_current}"
