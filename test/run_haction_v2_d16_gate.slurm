#!/usr/bin/env bash
#SBATCH --job-name=tk_hact_v2_d16
#SBATCH --partition=amd_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/haction_v2_d16_gate/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/haction_v2_d16_gate/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly baseline_engine=${migration_root}/engine_leftedge_compact_pingpong_v1_20260919
readonly candidate_engine=${migration_root}/engine_leftedge_haction_v2_20260919
readonly converter_root=${migration_root}/converter_anchor_v2_20260918
readonly source_root=${migration_root}/small_mpi_regression/source_d16
readonly driver=${candidate_engine}/test/four_direction_snapshot.jl
readonly run_root=${migration_root}/haction_v2_d16_gate/run_${SLURM_JOB_ID}
readonly shm_parent=/dev/shm/${USER}/dmrg_hact_v2_d16_${SLURM_JOB_ID}
readonly local_parent=/tmp/${USER}/dmrg_hact_v2_d16_${SLURM_JOB_ID}
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test -d "${baseline_engine}"
test -d "${candidate_engine}"
test -d "${source_root}"
test ! -e "${run_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${run_root}" "${shm_parent}" "${local_parent}"
cleanup_owned=true

export JULIA_DEPOT_PATH=${migration_root}/depot_final:/public4/home/sc56578/.julia
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

sha256sum "${candidate_engine}/MPS/TN.jl" "${candidate_engine}/MPS.jl" \
  "${candidate_engine}/test/mpi_qn_block_reduce.jl" \
  "${candidate_engine}/test/environment_cache_invalidation_mpi.jl" \
  "${candidate_engine}/test/four_direction_snapshot.jl" \
  "${candidate_engine}/test/run_haction_v2_d16_gate.slurm" \
  >"${run_root}/candidate_sha256.txt"
sha256sum "${baseline_engine}/MPS/TN.jl" "${baseline_engine}/MPS.jl" \
  >"${run_root}/baseline_sha256.txt"

cd "${candidate_engine}"
julia --project=. -e '
    using MPI
    launcher=MPI.mpiexec()
    child=`julia --project=. test/mpi_qn_block_reduce.jl`
    run(`$launcher -n 4 $child`)
' >"${run_root}/packed_reduce_test.out" 2>"${run_root}/packed_reduce_test.err"

export ENVIRONMENT_CACHE_TEST_ROOT=${shm_parent}/cache_test
export ENVIRONMENT_STORE_OWNER=${SLURM_JOB_ID}_cache_test
julia --project=. -e '
    using MPI
    launcher=MPI.mpiexec()
    child=`julia --project=. test/environment_cache_invalidation_mpi.jl`
    run(`$launcher -n 4 $child`)
' >"${run_root}/cache_test.out" 2>"${run_root}/cache_test.err"
grep -q 'environment_cache_invalidation_mpi=PASS ranks=4' "${run_root}/cache_test.out"
unset ENVIRONMENT_CACHE_TEST_ROOT ENVIRONMENT_STORE_OWNER

export model=Square_Hubbard_U1_U1_OBC_LeftEdge
export Lx=3 L2=2 Ne=6 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=16 INPUT_BOND_DIMENSION=16 STOP_BOND_DIMENSION=16
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR=${source_root}
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=4 ENV_VERIFY_WRITES=true PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=0 ENVIRONMENT_WINDOW_ANCHORS=false
export ENV_STABLE_TERM_OWNERS=true ENV_WINDOW_TRANSIENT_GB=0
export ENV_TIERED_PINGPONG=false ENV_PRIMARY_CAP_GB=0
export H_ACTION_SCHEDULER=static_all_ranks
export DIRECTION_PASSES=4 DIRECTION_DRIVER=${driver}

run_case() {
  local label=$1 engine=$2 reduce_mode=$3
  export ENGINE_ROOT=${engine}
  export MPI_TENSORMAP_REDUCE_MODE=${reduce_mode}
  export OUTPUT_MPS_DIR=${run_root}/${label}_mps
  export DIRECTION_SNAPSHOT_ROOT=${run_root}/${label}_snapshots
  export ENVIRONMENT_TMP_ROOT=${shm_parent}/${label}
  export ENVIRONMENT_SPILL_ROOT=${local_parent}/${label}_spill
  export ENVIRONMENT_STORE_OWNER=${SLURM_JOB_ID}_${label}
  export RUN_TAG=hact_v2_${label}_${SLURM_JOB_ID}
  mkdir -p "${DIRECTION_SNAPSHOT_ROOT}"
  cd "${engine}"
  /usr/bin/time -v julia --project=. -e '
      using MPI
      launcher=MPI.mpiexec()
      driver=ENV["DIRECTION_DRIVER"]
      child=`julia --project=. $driver`
      run(`$launcher -n 4 $child`)
  ' >"${run_root}/${label}.out" 2>"${run_root}/${label}.err"
  grep -q 'four_direction_snapshot=PASS passes=4' "${run_root}/${label}.out"
  test "$(wc -l <"${DIRECTION_SNAPSHOT_ROOT}/directions.tsv" | awk '{print $1}')" -eq 4
  ! grep -Eq 'Tensor file not found|SpaceMismatch|Ambiguous environment backing|BACKING_ERROR' \
    "${run_root}/${label}.out" "${run_root}/${label}.err"
}

run_case baseline "${baseline_engine}" legacy
run_case candidate_legacy_reduce "${candidate_engine}" legacy
run_case candidate_packed "${candidate_engine}" packed

compare_case() {
  local candidate=$1
  for pass in 1 2 3 4; do
    local direction=forward
    test $((pass%2)) -eq 0 && direction=reverse
    cd "${converter_root}"
    julia --project=. compare_tensorkit_mps.jl \
      "${run_root}/baseline_snapshots/pass_${pass}_${direction}" \
      "${run_root}/${candidate}_snapshots/pass_${pass}_${direction}" \
      | tee "${run_root}/baseline_vs_${candidate}_pass_${pass}.txt"
    awk -F= '$1=="normalized_overlap" {found=1;ok=($2>=1-1e-10)} END {exit !(found&&ok)}' \
      "${run_root}/baseline_vs_${candidate}_pass_${pass}.txt"
    awk -F= '$1=="fidelity_per_site" {found=1;ok=($2>=1-1e-12)} END {exit !(found&&ok)}' \
      "${run_root}/baseline_vs_${candidate}_pass_${pass}.txt"
    test "$(awk -F= '$1=="site_spaces_match" {print $2}' \
      "${run_root}/baseline_vs_${candidate}_pass_${pass}.txt")" = true
  done
  paste "${run_root}/baseline_snapshots/directions.tsv" \
    "${run_root}/${candidate}_snapshots/directions.tsv" | awk '{
      de=$3-$8;if(de<0)de=-de;dt=$4-$9;if(dt<0)dt=-dt;
      if(de>1e-9 || dt>1e-12)exit 1
    }'
}

compare_case candidate_legacy_reduce
compare_case candidate_packed
test ! -e "${shm_parent}/baseline"
test ! -e "${shm_parent}/candidate_legacy_reduce"
test ! -e "${shm_parent}/candidate_packed"
printf 'haction_v2_d16_gate=PASS\n' | tee "${run_root}/PASS"
