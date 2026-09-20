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
readonly gate_root="${migration_root}/directional_restart_gate/run_${SLURM_JOB_ID}"
readonly direct_output="${gate_root}/direct_two_pass"
readonly forward_output="${gate_root}/split_forward"
readonly reverse_output="${gate_root}/split_reverse"

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
export SVD_MODE=mpi KRYLOV_DIM=3 KRYLOV_TOL=1e-14
export TRUNCATION_CUTOFF=1e-9 ENERGY_ONLY=false

run_engine() {
  export START_MPS_DIR=$1
  export OUTPUT_MPS_DIR=$2
  export SWEEP_DIMS=$3
  export FIRST_PASS_REVERSE=$4
  export RUN_TAG=$5
  julia --project=. -e '
      using MPI
      nranks=parse(Int,ENV["SLURM_NTASKS"])
      launcher=MPI.mpiexec()
      child=`julia --project=. start.jl`
      run(`$launcher -n $nranks $child`)
  '
}

run_engine "${source_root}" "${direct_output}" 16,16 false \
  "restart_direct_${SLURM_JOB_ID}"
run_engine "${source_root}" "${forward_output}" 16 false \
  "restart_forward_${SLURM_JOB_ID}"
run_engine "${forward_output}" "${reverse_output}" 16 true \
  "restart_reverse_${SLURM_JOB_ID}"

direct_current="${direct_output}/$(tr -d '\n' < "${direct_output}/CURRENT")"
reverse_current="${reverse_output}/$(tr -d '\n' < "${reverse_output}/CURRENT")"
cd "${migration_root}/converter"
julia --project=. compare_tensorkit_mps.jl \
  "${direct_current}" "${reverse_current}" | tee "${gate_root}/comparison.txt"

overlap=$(awk -F= '$1 == "normalized_overlap" {print $2}' "${gate_root}/comparison.txt")
awk -v value="${overlap}" 'BEGIN { exit !(value >= 1.0 - 1.0e-11) }'
printf 'directional_restart_gate=PASS\noverlap=%s\n' "${overlap}"
