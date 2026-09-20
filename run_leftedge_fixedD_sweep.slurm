#!/usr/bin/env bash
#SBATCH --partition=amd_1T
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=800G
#SBATCH --time=48:00:00

set -euo pipefail

: "${CASE_LABEL:?CASE_LABEL is required}"
: "${NE:?NE is required}"
: "${BOND_D:?BOND_D is required}"
: "${SWEEP_DIMS:?SWEEP_DIMS is required}"
: "${START_MPS_DIR:?START_MPS_DIR is required}"
: "${OUTPUT_MPS_DIR:?OUTPUT_MPS_DIR is required}"

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_20260917"

test -d "${START_MPS_DIR}"
test ! -e "${OUTPUT_MPS_DIR}"
mkdir -p "$(dirname "${OUTPUT_MPS_DIR}")"
cd "${engine_root}"

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1
export OMP_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export model=Square_Hubbard_U1_U1_OBC_LeftEdge
export Lx=32
export L2=6
export Ne="${NE}"
export SZ2=0
export U=12.0
export t=1.0
export ty=1.0
export PINNING=left_edge_af
export PINNING_STRENGTH=0.1
export BOND_DIMENSION="${BOND_D}"
export INPUT_BOND_DIMENSION="${BOND_D}"
export STOP_BOND_DIMENSION="${BOND_D}"
export SWEEP_DIMS
export START_MPS_DIR
export OUTPUT_MPS_DIR
export RUN_TAG="leftedge_${CASE_LABEL}_D${BOND_D}_${SLURM_JOB_ID}"
export SVD_MODE=mpi
export KRYLOV_DIM=3
export KRYLOV_TOL=1e-14
export TRUNCATION_CUTOFF=1e-9
export ENERGY_ONLY=false
export FIRST_PASS_REVERSE="${FIRST_PASS_REVERSE:-false}"

printf 'case=%s\nNe=%s\nD=%s\nsweep_dims=%s\nfirst_pass_reverse=%s\nstart=%s\noutput=%s\nranks=%s\n' \
  "${CASE_LABEL}" "${Ne}" "${BOND_D}" "${SWEEP_DIMS}" \
  "${FIRST_PASS_REVERSE}" "${START_MPS_DIR}" "${OUTPUT_MPS_DIR}" "${SLURM_NTASKS}"
/usr/bin/time -v julia --project=. -e '
    using MPI
    nranks=parse(Int,ENV["SLURM_NTASKS"])
    launcher=MPI.mpiexec()
    child=`julia --project=. start.jl`
    run(`$launcher -n $nranks $child`)
'

test -f "${OUTPUT_MPS_DIR}/CURRENT"
test -f "${OUTPUT_MPS_DIR}/PREVIOUS"
