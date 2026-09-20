#!/usr/bin/env bash
#SBATCH --job-name=tk_bound_d16
#SBATCH --partition=amd_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_bounded_d16_gate/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/hybrid_bounded_d16_gate/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root=${migration_root}/engine_leftedge_hybrid_bounded_v1_20260920
readonly converter_root=${migration_root}/converter_anchor_v2_20260918
readonly source_root=${migration_root}/small_mpi_regression/source_d16
readonly affinity_pass=${migration_root}/hybrid_thread_affinity_smoke/run_12175622/PASS
readonly gate_root=${migration_root}/hybrid_bounded_d16_gate
readonly batch_script=${gate_root}/run_hybrid_bounded_d16_gate.slurm
readonly full_driver=${gate_root}/single_direction_snapshot.jl
readonly bounded_driver=${gate_root}/bounded_direction_snapshot.jl
readonly run_root=${gate_root}/run_${SLURM_JOB_ID}
readonly shm_parent=/dev/shm/${USER}/dmrg_bound_d16_${SLURM_JOB_ID}
readonly tmp_parent=/tmp/${USER}/dmrg_bound_d16_${SLURM_JOB_ID}
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${tmp_parent}"
  fi
}
trap cleanup EXIT INT TERM

test "$(cat "${affinity_pass}")" = 'hybrid_thread_affinity_smoke=PASS'
test -d "${source_root}"
test ! -e "${run_root}"
test ! -e "${shm_parent}"
test ! -e "${tmp_parent}"
mkdir -p "${run_root}" "${shm_parent}" "${tmp_parent}"
cleanup_owned=true

export JULIA_DEPOT_PATH=${migration_root}/depot_final:/public4/home/sc56578/.julia
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 TENSORKIT_TRANSFORMER_THREADS=1
export OMP_PLACES=cores OMP_PROC_BIND=close MKL_DYNAMIC=false
export model=${physical_label}
export Lx=3 L2=2 Ne=6 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=16 INPUT_BOND_DIMENSION=16 STOP_BOND_DIMENSION=16
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi SWEEP_DIMS=16 FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR=${source_root}
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0
export ENV_PRIMARY_RESERVE_GB=4 ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=0 ENVIRONMENT_WINDOW_ANCHORS=false
export ENV_STABLE_TERM_OWNERS=true ENV_WINDOW_TRANSIENT_GB=0
export ENV_TIERED_PINGPONG=false ENV_PRIMARY_CAP_GB=0
export HOPPING_CHANNEL_MODE=split_pruned H_ACTION_PIN_CACHE_GB=0
export H_ACTION_SCHEDULER=static_all_ranks MPI_TENSORMAP_REDUCE_MODE=packed
export ENGINE_ROOT=${engine_root}

sha256sum "${engine_root}/MPS/TN.jl" "${engine_root}/MPS.jl" \
  "${engine_root}/model/Hubbard_OBC_LeftEdge_U1_U1_core.jl" \
  "${batch_script}" "${full_driver}" "${bounded_driver}" \
  >"${run_root}/sha256.txt"
find "${source_root}" -maxdepth 2 -type f -printf '%P\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_before.tsv"

run_case() {
  local label=$1 threads=$2 driver=$3 bounded=$4
  local output=${run_root}/${label}_mps
  local work=${run_root}/${label}_work
  export OMP_THREADS=${threads} OMP_NUM_THREADS=${threads}
  export MKL_NUM_THREADS=${threads} OPENBLAS_NUM_THREADS=${threads}
  export OUTPUT_MPS_DIR=${output}
  export ENVIRONMENT_TMP_ROOT=${shm_parent}/${label}
  export ENVIRONMENT_SPILL_ROOT=${tmp_parent}/${label}_spill
  export ENVIRONMENT_STORE_OWNER=${SLURM_JOB_ID}_${label}
  export RUN_TAG=bound_d16_${label}_${SLURM_JOB_ID}
  if test "${bounded}" -gt 0; then
    export BENCHMARK_MAX_BONDS=${bounded}
  else
    unset BENCHMARK_MAX_BONDS || true
  fi
  mkdir -p "${work}/out/${model}" "${work}/out_short/${model}" \
    "${work}/time/${model}" "${work}/EE/${model}"
  cd "${work}"
  srun --exclusive --mpi=pmi2 --ntasks=4 --cpus-per-task=16 --cpu-bind=cores \
    julia --project="${engine_root}" "${driver}" \
    >"${run_root}/${label}.out" 2>"${run_root}/${label}.err"
  grep -q "THREAD_CONFIGURATION julia=1 tensor_transformer=1 blas=${threads}" \
    "${run_root}/${label}.out"
  test -f "${output}/CURRENT" && test -f "${output}/PREVIOUS"
  current=${output}/$(tr -d '\n' <"${output}/CURRENT")
  test -f "${current}/checkpoint_manifest.toml"
  grep -q 'complete = true' "${current}/checkpoint_manifest.toml"
  if test "${bounded}" -gt 0; then
    grep -q "BOUNDED_SWEEP_COMPLETE direction=forward processed_bonds=${bounded} total_bonds=5" \
      "${run_root}/${label}.out"
    grep -q "bounded_direction_snapshot=PASS direction=forward bonds=${bounded}" \
      "${run_root}/${label}.out"
    grep -q 'partial_direction = true' "${current}/checkpoint_manifest.toml"
    grep -q "processed_bonds = ${bounded}" "${current}/checkpoint_manifest.toml"
  else
    grep -q 'single_direction_snapshot=PASS direction=forward' "${run_root}/${label}.out"
    grep -q 'partial_direction = false' "${current}/checkpoint_manifest.toml"
  fi
  ! grep -Eq 'SpaceMismatch|BACKING_ERROR|OutOfMemoryError|Tensor file not found' \
    "${run_root}/${label}.out" "${run_root}/${label}.err"
}

# The full case proves that the default path remains a complete direction.
# The two bounded cases prove deterministic two-bond stopping and BLAS
# invariance before any D5000 partial-direction benchmark is authorized.
run_case full_default 1 "${full_driver}" 0
run_case bounded_blas1 1 "${bounded_driver}" 2
run_case bounded_blas16 16 "${bounded_driver}" 2

current_generation() {
  local checkpoint=$1
  printf '%s/%s' "${checkpoint}" "$(tr -d '\n' <"${checkpoint}/CURRENT")"
}
cd "${converter_root}"
julia --project=. compare_tensorkit_mps.jl \
  "$(current_generation "${run_root}/bounded_blas1_mps")" \
  "$(current_generation "${run_root}/bounded_blas16_mps")" \
  >"${run_root}/bounded_blas1_vs_blas16.txt"
awk -F= '$1=="normalized_overlap"{found=1;ok=($2>=1-1e-12)}END{exit !(found&&ok)}' \
  "${run_root}/bounded_blas1_vs_blas16.txt"
awk -F= '$1=="fidelity_per_site"{found=1;ok=($2>=1-1e-14)}END{exit !(found&&ok)}' \
  "${run_root}/bounded_blas1_vs_blas16.txt"
test "$(awk -F= '$1=="site_spaces_match"{print $2}' \
  "${run_root}/bounded_blas1_vs_blas16.txt")" = true

find "${source_root}" -maxdepth 2 -type f -printf '%P\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_after.tsv"
cmp "${run_root}/source_inventory_before.tsv" "${run_root}/source_inventory_after.tsv"
for label in full_default bounded_blas1 bounded_blas16; do
  actual_primary=$(awk -F= '$1=="ENVIRONMENT_STORE_ROOT"{print $2;exit}' \
    "${run_root}/${label}.out")
  actual_spill=$(awk -F= '$1=="ENVIRONMENT_STORE_SPILL_ROOT"{print $2;exit}' \
    "${run_root}/${label}.out")
  test -n "${actual_primary}" && test -n "${actual_spill}"
  test ! -e "${actual_primary}" && test ! -e "${actual_spill}"
done

printf 'numerical=PASS\n' >"${run_root}/NUMERICAL_PASS"
printf 'hybrid_bounded_d16_gate=PASS\n' >"${run_root}/PASS"
