#!/usr/bin/env bash
#SBATCH --partition=amd_test
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_20260917"
readonly source_root="${migration_root}/small_mpi_regression/source_d16"
readonly gate_root="${migration_root}/contiguous_two_pass_gate/run_${SLURM_JOB_ID}"
readonly serial_output="${gate_root}/serial"
readonly mpi_output="${gate_root}/mpi_r8"

test -d "${source_root}"
test ! -e "${gate_root}"
mkdir -p "${gate_root}"
cd "${engine_root}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model=Square_Hubbard_U1_U1_OBC_LeftEdge
export Lx=3 L2=2 Ne=6 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=16 INPUT_BOND_DIMENSION=16 STOP_BOND_DIMENSION=16
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14
export TRUNCATION_CUTOFF=1e-9 ENERGY_ONLY=false
export SWEEP_DIMS=16:16 FIRST_PASS_REVERSE=false

export START_MPS_DIR="${source_root}"
export OUTPUT_MPS_DIR="${serial_output}"
export RUN_TAG="two_pass_serial_${SLURM_JOB_ID}"
export SVD_MODE=serial
julia --project=. start.jl

export OUTPUT_MPS_DIR="${mpi_output}"
export RUN_TAG="two_pass_mpi8_${SLURM_JOB_ID}"
export SVD_MODE=mpi
julia --project=. -e '
    using MPI
    nranks=parse(Int,ENV["SLURM_NTASKS"])
    launcher=MPI.mpiexec()
    child=`julia --project=. start.jl`
    run(`$launcher -n $nranks $child`)
'

test -f "${serial_output}/CURRENT"
test -f "${serial_output}/PREVIOUS"
test -f "${mpi_output}/CURRENT"
test -f "${mpi_output}/PREVIOUS"
serial_current="${serial_output}/$(tr -d '\n' < "${serial_output}/CURRENT")"
mpi_current="${mpi_output}/$(tr -d '\n' < "${mpi_output}/CURRENT")"

cd "${migration_root}/converter"
julia --project=. compare_tensorkit_mps.jl \
  "${serial_current}" "${mpi_current}" | tee "${gate_root}/comparison.txt"

overlap=$(awk -F= '$1 == "normalized_overlap" {print $2}' "${gate_root}/comparison.txt")
fidelity=$(awk -F= '$1 == "fidelity_per_site" {print $2}' "${gate_root}/comparison.txt")
awk -v value="${overlap}" 'BEGIN { exit !(value >= 1.0 - 1.0e-11) }'
awk -v value="${fidelity}" 'BEGIN { exit !(value >= 1.0 - 1.0e-12) }'
printf 'contiguous_two_pass_gate=PASS\noverlap=%s\nfidelity_per_site=%s\n' \
  "${overlap}" "${fidelity}"

