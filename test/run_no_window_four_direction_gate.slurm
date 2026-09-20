#!/usr/bin/env bash
#SBATCH --job-name=tk_nw4_gate
#SBATCH --partition=amd_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/no_window_four_direction_gate/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/no_window_four_direction_gate/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v3_20260919"
readonly source_root="${migration_root}/small_mpi_regression/source_d16"
readonly run_root="${migration_root}/no_window_four_direction_gate/run_${SLURM_JOB_ID}"
readonly output_root="${run_root}/mps"
readonly shm_parent="/dev/shm/${USER}/dmrg_nw4_gate_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_nw4_gate_${SLURM_JOB_ID}"
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test -d "${source_root}"
test ! -e "${run_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${run_root}" "${shm_parent}" "${local_parent}"
cleanup_owned=true

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model=Square_Hubbard_U1_U1_OBC_LeftEdge
export Lx=3 L2=2 Ne=6 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=16 INPUT_BOND_DIMENSION=16 STOP_BOND_DIMENSION=16
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=16:16:16:16
export FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR="${source_root}"
export OUTPUT_MPS_DIR="${output_root}"
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=4 ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=0
export ENVIRONMENT_WINDOW_ANCHORS=false
export ENV_STABLE_TERM_OWNERS=false
export ENV_WINDOW_TRANSIENT_GB=0
export ENVIRONMENT_TMP_ROOT="${shm_parent}/environment"
export ENVIRONMENT_SPILL_ROOT="${local_parent}/environment_spill"
export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_nw4_gate"
export RUN_TAG="nw4_gate_${SLURM_JOB_ID}"

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl \
  start.jl test/run_no_window_four_direction_gate.slurm \
  >"${run_root}/engine_sha256.txt"

julia --project=. -e '
    using MPI
    launcher=MPI.mpiexec()
    child=`julia --project=. start.jl`
    run(`$launcher -n 4 $child`)
' >"${run_root}/engine.out" 2>"${run_root}/engine.err"

grep -Eq 'ENVIRONMENT_WINDOW_SELECTION D=16 window=0 configured=0' "${run_root}/engine.out"
test "$(grep -c 'SEGMENT_TIMING label=initial-full-chain name=initial_total' "${run_root}/engine.out")" -eq 1
for spec in '1-forward' '2-reverse' '3-forward' '4-reverse'; do
  grep -q "SEGMENT_TIMING label=sweep-${spec} name=sweep_total" "${run_root}/engine.out"
done
test "$(grep -c 'SEGMENT_TIMING label=sweep-.* name=sweep_total' "${run_root}/engine.out")" -eq 4
! grep -Eq 'Tensor file not found|SpaceMismatch|missing_drops = [1-9][0-9]*|BACKING_ERROR' \
  "${run_root}/engine.out" "${run_root}/engine.err"

test -f "${output_root}/CURRENT"
test -f "${output_root}/PREVIOUS"
for pointer in CURRENT PREVIOUS; do
  generation="${output_root}/$(tr -d '\n' < "${output_root}/${pointer}")"
  test -f "${generation}/checkpoint_manifest.toml"
  test "$(find "${generation}" -maxdepth 1 -type f \( -name '*.jld2' -o -name '*.bin' \) | wc -l | awk '{print $1}')" -eq 6
  grep -q 'complete = true' "${generation}/checkpoint_manifest.toml"
done
current="${output_root}/$(tr -d '\n' < "${output_root}/CURRENT")"
previous="${output_root}/$(tr -d '\n' < "${output_root}/PREVIOUS")"
grep -q 'reverse = true' "${current}/checkpoint_manifest.toml"
grep -q 'reverse = false' "${previous}/checkpoint_manifest.toml"

actual_primary=$(grep -m1 '^ENVIRONMENT_STORE_ROOT=' "${run_root}/engine.out" | cut -d= -f2-)
actual_spill=$(grep -m1 '^ENVIRONMENT_STORE_SPILL_ROOT=' "${run_root}/engine.out" | cut -d= -f2-)
test -n "${actual_primary}" && test -n "${actual_spill}"
test ! -e "${actual_primary}" && test ! -e "${actual_spill}"
printf 'no_window_four_direction_gate=PASS\n' | tee "${run_root}/PASS"
