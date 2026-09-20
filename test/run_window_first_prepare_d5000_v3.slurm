#!/usr/bin/env bash
#SBATCH --job-name=tk_firstwin_D5k_v3
#SBATCH --partition=amd_256
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=234G
#SBATCH --time=04:00:00
#SBATCH --output=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_first_prepare_d5000_v3/logs/%x.%j.out
#SBATCH --error=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration/window_first_prepare_d5000_v3/logs/%x.%j.err

set -euo pipefail

readonly migration_root=/public4/home/sc56578/zyc/hubbard_flux/new_engine_migration
readonly engine_root="${migration_root}/engine_leftedge_anchor_v3_20260919"
readonly source_root="${migration_root}/leftedge_D5000_migration_20260917/half/itensor_converted"
readonly source_current=flat-itensor-converted
readonly source_generation="${source_root}"
readonly run_root="${migration_root}/window_first_prepare_d5000_v3/run_${SLURM_JOB_ID}"
readonly shm_parent="/dev/shm/${USER}/dmrg_firstwin_D5k_v3_${SLURM_JOB_ID}"
readonly local_parent="/tmp/${USER}/dmrg_firstwin_D5k_v3_${SLURM_JOB_ID}"
readonly physical_label=Square_Hubbard_U1_U1_OBC_LeftEdge
cleanup_owned=false

cleanup() {
  if test "${cleanup_owned}" = true; then
    rm -rf -- "${shm_parent}" "${local_parent}"
  fi
}
trap cleanup EXIT INT TERM

test -d "${source_generation}"
test -f "${source_generation}/1.jld2"
test -f "${source_generation}/192.jld2"
test ! -e "${run_root}"
test ! -e "${shm_parent}"
test ! -e "${local_parent}"
mkdir -p "${run_root}" "${shm_parent}" "${local_parent}"
cleanup_owned=true

export JULIA_DEPOT_PATH="${migration_root}/depot_final:/public4/home/sc56578/.julia"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OMP_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export model="${physical_label}"
export Lx=32 L2=6 Ne=192 SZ2=0 U=12.0 t=1.0 ty=1.0
export PINNING=left_edge_af PINNING_STRENGTH=0.1
export BOND_DIMENSION=5000 INPUT_BOND_DIMENSION=5000 STOP_BOND_DIMENSION=5000
export SVD_MODE=mpi KRYLOV_DIM=3 KRYLOV_TOL=1e-14 TRUNCATION_CUTOFF=1e-9
export FIRST_PASS_REVERSE=false ENERGY_ONLY=false
export SOURCE_GENERATION="${source_generation}"
export ENV_ROLLING_RELEASE=true ENV_LOCAL_CACHE_GB=0.125
export ENV_PRIMARY_RESERVE_GB=16 ENV_VERIFY_WRITES=false PROFILE_SEGMENTS=true
export ENVIRONMENT_WINDOW_SITES=schedule
export ENVIRONMENT_WINDOW_SCHEDULE=1000:32,5000:32,9000:24,12000:20,20000:16
export ENVIRONMENT_WINDOW_ANCHORS=true
export ENV_WINDOW_VERIFY_OWNERSHIP=true
export ENV_WINDOW_PROGRESS_SITES=8
export ENV_STABLE_TERM_OWNERS=true
export ENV_WINDOW_TRANSIENT_GB=24
export RUN_TAG="firstwin_D5k_v3_${SLURM_JOB_ID}"
export ENVIRONMENT_TMP_ROOT="${shm_parent}/environment"
export ENVIRONMENT_SPILL_ROOT="${local_parent}/environment_spill"
export ENVIRONMENT_STORE_OWNER="${SLURM_JOB_ID}_firstwin_D5k_v3"

cd "${engine_root}"
sha256sum MPS/TN.jl MPS.jl model/Square_Hubbard_U1_U1_OBC_LeftEdge.jl \
  test/window_first_prepare_benchmark.jl test/run_window_first_prepare_d5000_v3.slurm \
  >"${run_root}/engine_sha256.txt"
sha256sum "${source_root}/conversion.toml" "${source_root}/1.jld2" \
  "${source_root}/96.jld2" "${source_root}/192.jld2" \
  >"${run_root}/source_sha256_before.txt"
find "${source_root}" -maxdepth 1 -type f -printf '%f %s %T@\n' | sort \
  >"${run_root}/source_inventory_before.txt"
printf '%s\n' "${source_current}" >"${run_root}/source_CURRENT"

monitor_storage() {
  local watched_pid=$1 destination=$2
  printf 'unix_time primary_kib spill_kib shm_available_kib tmp_available_kib\n' >"${destination}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local a=0 b=0
    test ! -e "${shm_parent}" || a=$(du -sk "${shm_parent}" 2>/dev/null | awk '{print $1+0}')
    test ! -e "${local_parent}" || b=$(du -sk "${local_parent}" 2>/dev/null | awk '{print $1+0}')
    printf '%s %s %s %s %s\n' "$(date +%s)" "${a}" "${b}" \
      "$(df -Pk /dev/shm | awk 'NR==2{print $4+0}')" \
      "$(df -Pk /tmp | awk 'NR==2{print $4+0}')" >>"${destination}"
    sleep 10
  done
}

started=$(date +%s)
/usr/bin/time -v julia --project=. -e '
    using MPI
    launcher=MPI.mpiexec()
    child=`julia --project=. test/window_first_prepare_benchmark.jl`
    run(`$launcher -n 4 $child`)
' >"${run_root}/engine.out" 2>"${run_root}/engine.err" &
pid=$!
monitor_storage "${pid}" "${run_root}/storage.tsv" &
monitor_pid=$!
status=0
wait "${pid}" || status=$?
wait "${monitor_pid}" || true
finished=$(date +%s)
printf '%s\n' "$((finished-started))" >"${run_root}/case_wall_seconds"
test "${status}" -eq 0

grep -Eq 'FIRST_WINDOW_RESULT status=PASS.*bonds=1:32.*sites=192 D=5000 ranks=4.*environment_site_updates=190 anchor_snapshots=5' \
  "${run_root}/engine.out"
grep -Eq 'ENVIRONMENT_WINDOW_SELECTION D=5000 window=32 configured=schedule' "${run_root}/engine.out"
grep -Eq 'WINDOW_TRANSIENT_STATS direction=right.*hits=[1-9][0-9]* backing_fallbacks=0' "${run_root}/engine.out"
! grep -Eq 'missing_drops = [1-9][0-9]*|Tensor file not found|BACKING_ERROR' \
  "${run_root}/engine.out" "${run_root}/engine.err"
for segment in window_prepare window_prepass_mini window_prepass_combined \
    window_retained_publish window_anchor_snapshot checkpoint_site_read; do
  grep -q "SEGMENT_TIMING label=first-window-forward name=${segment}" "${run_root}/engine.out"
done

actual_primary=$(grep -m1 '^ENVIRONMENT_STORE_ROOT=' "${run_root}/engine.out" | cut -d= -f2-)
actual_spill=$(grep -m1 '^ENVIRONMENT_STORE_SPILL_ROOT=' "${run_root}/engine.out" | cut -d= -f2-)
test -n "${actual_primary}" && test -n "${actual_spill}"
test ! -e "${actual_primary}" && test ! -e "${actual_spill}"
sha256sum "${source_root}/conversion.toml" "${source_root}/1.jld2" \
  "${source_root}/96.jld2" "${source_root}/192.jld2" \
  >"${run_root}/source_sha256_after.txt"
find "${source_root}" -maxdepth 1 -type f -printf '%f %s %T@\n' | sort \
  >"${run_root}/source_inventory_after.txt"
cmp "${run_root}/source_sha256_before.txt" "${run_root}/source_sha256_after.txt"
cmp "${run_root}/source_inventory_before.txt" "${run_root}/source_inventory_after.txt"
test ! -e "${run_root}/mps"

printf 'window_first_prepare_d5000_v3=PASS\n' | tee "${run_root}/PASS"
