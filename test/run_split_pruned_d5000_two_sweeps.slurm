#!/usr/bin/env bash
#SBATCH --job-name=tkD5k_split
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=234G
#SBATCH --time=12:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/leftedge_D5000_split_pruned_2sweeps_20260919/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/leftedge_D5000_split_pruned_2sweeps_20260919/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root=${migration_root}/engine_leftedge_haction_v6_20260919
readonly d1000_pass=${migration_root}/haction_v6_d1000_symmetric/run_12171185/PASS
readonly production_root=${migration_root}/leftedge_D5000_split_pruned_2sweeps_20260919
readonly batch_script=${production_root}/run_split_pruned_d5000_two_sweeps.slurm
readonly direction_driver=${production_root}/four_direction_rolling.jl
readonly case_name=${CASE:?CASE=half or doped is required}
case "${case_name}" in
  half)
    readonly particle_number=192
    readonly source_root=${migration_root}/leftedge_D5000_migration_20260917/half/itensor_converted
    ;;
  doped)
    readonly particle_number=180
    readonly source_root=${migration_root}/leftedge_D5000_migration_20260917/doped_n09375/itensor_converted
    ;;
  *)
    echo "CASE must be half or doped, got ${case_name}" >&2
    exit 2
    ;;
esac
readonly run_root=${production_root}/${case_name}/job_${SLURM_JOB_ID}
readonly output_root=${run_root}/mps
readonly work_root=${run_root}/work
readonly shm_parent=/dev/shm/${USER}/dmrg_D5k_split_${case_name}_${SLURM_JOB_ID}
readonly local_parent=/tmp/${USER}/dmrg_D5k_split_${case_name}_${SLURM_JOB_ID}
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test "$(cat "${d1000_pass}")" = 'haction_v6_d1000_symmetric=PASS'
test -f "${source_root}/1.jld2"
test -f "${source_root}/96.jld2"
test -f "${source_root}/192.jld2"
test -f "${source_root}/conversion.toml"
test "$(find "${source_root}" -maxdepth 1 -type f | wc -l | awk '{print $1}')" -eq 193
test ! -e "${run_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${run_root}" "${work_root}/out/${physical_label}" \
  "${work_root}/out_short/${physical_label}" "${work_root}/time/${physical_label}" \
  "${work_root}/EE/${physical_label}" "${shm_parent}" "${local_parent}"
cleanup_owned=true

# The selected D1000 representation reduced full no-window backing from
# 15.65 GiB to 2.25 GiB.  Require two independent node-local tiers; cap tmpfs
# so active tensors retain most of the 234 GiB allocation.
test "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" -ge 83886080
test "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" -ge 83886080
test "$(df -Pk "${migration_root}" | awk 'NR==2{print $4+0}')" -ge 41943040

export JULIA_DEPOT_PATH=${migration_root}/depot_final:/public4/home/sc56578/.julia
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 TENSORKIT_TRANSFORMER_THREADS=1
export OMP_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model=${physical_label}
export Lx=32 L2=6 Ne=${particle_number} SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=5000 INPUT_BOND_DIMENSION=5000 STOP_BOND_DIMENSION=5000
export KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export SVD_MODE=mpi FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export START_MPS_DIR=${source_root}
export OUTPUT_MPS_DIR=${output_root}
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=32 ENV_PRIMARY_CAP_GB=64
export ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=0 ENVIRONMENT_WINDOW_ANCHORS=false
export ENV_STABLE_TERM_OWNERS=true ENV_WINDOW_TRANSIENT_GB=0
export ENV_TIERED_PINGPONG=false
export ENVIRONMENT_TMP_ROOT=${shm_parent}/environment
export ENVIRONMENT_SPILL_ROOT=${local_parent}/environment_spill
export ENVIRONMENT_STORE_OWNER=${SLURM_JOB_ID}_D5k_split_${case_name}
export HOPPING_CHANNEL_MODE=split_pruned
export H_ACTION_PIN_CACHE_GB=0
export H_ACTION_SCHEDULER=static_all_ranks
export MPI_TENSORMAP_REDUCE_MODE=packed
export ENGINE_ROOT=${engine_root}
export DIRECTION_DRIVER=${direction_driver}
export DIRECTION_RESULT_ROOT=${run_root}
export DIRECTION_PASSES=4
export RUN_TAG=D5k_split_${case_name}_${SLURM_JOB_ID}

sha256sum "${engine_root}/MPS/TN.jl" "${engine_root}/MPS.jl" \
  "${engine_root}/model/Hubbard_OBC_LeftEdge_U1_U1_core.jl" \
  "${engine_root}/model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl" \
  "${batch_script}" "${direction_driver}" >"${run_root}/engine_sha256.txt"
sha256sum "${source_root}/conversion.toml" "${source_root}/1.jld2" \
  "${source_root}/96.jld2" "${source_root}/192.jld2" \
  >"${run_root}/source_sha256_before.txt"
find "${source_root}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_before.tsv"
{
  printf 'job_id=%s\ncase=%s\ngeometry=32x6\nboundary=OBCxOBC\nordering=column-snake\n' "${SLURM_JOB_ID}" "${case_name}"
  printf 'U=12.0\nt=1.0\nty=1.0\nNe=%s\nSz2=0\npinning=left_edge_af\npinning_strength=0.1\n' "${particle_number}"
  printf 'D=5000\nfull_sweeps=2\ndirectional_passes=4\ndirections=forward,reverse,forward,reverse\n'
  printf 'window_sites=0\nhopping_channel_mode=split_pruned\nh_action_pin_cache_gb=0\n'
  printf 'h_action_scheduler=static_all_ranks\nmpi_reduce_mode=packed\n'
  printf 'ranks=4\njulia_threads=1\ntensorkit_transformer_threads=1\nblas_threads=1\n'
  printf 'krylov_dim=3\nkrylov_tol=1e-14\ntruncation_cutoff_squared_norm=1e-9\nsvd_mode=mpi\n'
  printf 'primary_cap_gb=64\nprimary_reserve_gb=32\nsource=%s\noutput=%s\n' "${source_root}" "${output_root}"
  printf 'early_review_after_forward_bonds=32\nearly_projected_two_sweep_limit_seconds=28800\n'
} >"${run_root}/launch_manifest.txt"
df -Pk /dev/shm /tmp "${migration_root}" >"${run_root}/storage_before_kib.txt"

monitor_resources() {
  local watched_pid=$1 destination=$2
  local valid_samples=0 failed_samples=0
  printf 'unix_time primary_kib spill_kib combined_kib shm_available_kib tmp_available_kib project_available_kib step_maxrss_raw\n' >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local primary_kib=0 spill_kib=0 maxrss=NA
    if test -e "${shm_parent}"; then
      primary_kib=$(du -sk "${shm_parent}" 2>/dev/null | awk '{print $1+0}') || { failed_samples=$((failed_samples+1)); sleep 30; continue; }
    fi
    if test -e "${local_parent}"; then
      spill_kib=$(du -sk "${local_parent}" 2>/dev/null | awk '{print $1+0}') || { failed_samples=$((failed_samples+1)); sleep 30; continue; }
    fi
    maxrss=$(sstat -j "${SLURM_JOB_ID}.0" --noheader --format=MaxRSS 2>/dev/null | awk 'NF{print $1;exit}')
    test -n "${maxrss}" || maxrss=NA
    printf '%s %s %s %s %s %s %s %s\n' "$(date +%s)" "${primary_kib}" "${spill_kib}" \
      "$((primary_kib+spill_kib))" "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" \
      "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" \
      "$(df -Pk "${migration_root}" | awk 'NR==2{print $4+0}')" "${maxrss}" >>"${destination}"
    valid_samples=$((valid_samples+1))
    sleep 30
  done
  printf '# monitor_summary valid_samples=%s failed_samples=%s\n' \
    "${valid_samples}" "${failed_samples}" >>"${destination}"
}

started=$(date +%s)
cd "${work_root}"
/usr/bin/time -v julia --project="${engine_root}" -e '
    using MPI
    launcher=MPI.mpiexec()
    project=ENV["ENGINE_ROOT"]
    driver=ENV["DIRECTION_DRIVER"]
    child=`julia --project=$project $driver`
    run(`$launcher -n 4 $child`)
' >"${run_root}/engine.out" 2>"${run_root}/engine.err" &
pid=$!
monitor_resources "${pid}" "${run_root}/resources.tsv" &
monitor_pid=$!
status=0
wait "${pid}" || status=$?
wait "${monitor_pid}" || true
finished=$(date +%s)
printf '%s\n' "$((finished-started))" >"${run_root}/case_wall_seconds"
test "${status}" -eq 0

grep -q 'four_direction_rolling=PASS passes=4' "${run_root}/engine.out"
grep -q 'ENVIRONMENT_WINDOW_SELECTION D=5000 window=0 configured=0' "${run_root}/engine.out"
grep -q 'H_ACTION_SCHEDULER=static_all_ranks' "${run_root}/engine.out"
grep -q 'CHANNEL_SCHEDULE_STATS.*raw_terms = 692.*haction_groups = 3694' "${run_root}/engine.out"
test "$(grep -c '^ROLLING_DIRECTION_RESULT pass=' "${run_root}/engine.out")" -eq 4
test "$(wc -l <"${run_root}/directions.tsv" | awk '{print $1}')" -eq 4
test "$(grep -c 'SEGMENT_TIMING label=initial-full-chain name=initial_total' "${run_root}/engine.out")" -eq 1
for spec in 1-forward 2-reverse 3-forward 4-reverse; do
  grep -q "SEGMENT_TIMING label=sweep-${spec} name=sweep_total" "${run_root}/engine.out"
done
! grep -Eq 'Tensor file not found|SpaceMismatch|missing_drops = [1-9][0-9]*|BACKING_ERROR|OutOfMemoryError' \
  "${run_root}/engine.out" "${run_root}/engine.err"

test -f "${output_root}/CURRENT" && test -f "${output_root}/PREVIOUS"
for pointer in CURRENT PREVIOUS; do
  generation=${output_root}/$(tr -d '\n' <"${output_root}/${pointer}")
  test -f "${generation}/checkpoint_manifest.toml"
  test "$(find "${generation}" -maxdepth 1 -type f | wc -l | awk '{print $1}')" -eq 193
  grep -q 'complete = true' "${generation}/checkpoint_manifest.toml"
done
current=${output_root}/$(tr -d '\n' <"${output_root}/CURRENT")
previous=${output_root}/$(tr -d '\n' <"${output_root}/PREVIOUS")
grep -q 'reverse = true' "${current}/checkpoint_manifest.toml"
grep -q 'reverse = false' "${previous}/checkpoint_manifest.toml"

sha256sum "${source_root}/conversion.toml" "${source_root}/1.jld2" \
  "${source_root}/96.jld2" "${source_root}/192.jld2" \
  >"${run_root}/source_sha256_after.txt"
find "${source_root}" -maxdepth 1 -type f -printf '%f\t%s\t%T@\n' \
  | LC_ALL=C sort >"${run_root}/source_inventory_after.tsv"
cmp "${run_root}/source_sha256_before.txt" "${run_root}/source_sha256_after.txt"
cmp "${run_root}/source_inventory_before.tsv" "${run_root}/source_inventory_after.tsv"
test ! -e "${shm_parent}/environment"
test ! -e "${local_parent}/environment_spill"
df -Pk /dev/shm /tmp "${migration_root}" >"${run_root}/storage_after_kib.txt"
printf 'split_pruned_d5000_two_sweeps=PASS\ncase=%s\ndirectional_passes=4\n' \
  "${case_name}" | tee "${run_root}/PASS"
